from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, EmailStr, Field
from server.database import get_db, hash_password, verify_password
from server.logger import logger
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import os

router = APIRouter(prefix="/auth")


# ----- Request Models -----

class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=6)

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    code: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class GoogleAuthRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr


# ----- OTP Email Sender -----

def send_otp_email(to_email: str, otp_code: str):
    """Send OTP via Gmail SMTP. Requires SMTP_EMAIL and SMTP_PASSWORD in .env"""
    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_email or not smtp_password:
        logger.warning("SMTP credentials not set. OTP will be logged instead.")
        logger.info(f"[DEV MODE] OTP for {to_email}: {otp_code}")
        return

    msg = MIMEMultipart()
    msg["From"] = smtp_email
    msg["To"] = to_email
    msg["Subject"] = "InsightsPop - Verify Your Account"

    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background: #0f1115; color: #f8fafc; padding: 40px;">
        <div style="max-width: 400px; margin: auto; background: rgba(26,29,36,0.9); padding: 32px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08);">
            <h1 style="background: linear-gradient(135deg, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">InsightsPop</h1>
            <p>Your verification code is:</p>
            <h2 style="letter-spacing: 8px; text-align: center; font-size: 32px; color: #8b5cf6; margin: 24px 0;">{otp_code}</h2>
            <p style="color: #94a3b8; font-size: 13px;">This code expires in 10 minutes. Do not share it with anyone.</p>
        </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, to_email, msg.as_string())
        logger.info(f"OTP email sent to {to_email}")
    except Exception as e:
        logger.exception(f"Failed to send OTP email: {e}")
        raise HTTPException(status_code=500, detail="Failed to send verification email")


# ----- Routes -----

@router.post("/register")
async def register(request: RegisterRequest, db=Depends(get_db)):
    cursor = db.cursor()

    # Check if user already exists
    cursor.execute("SELECT * FROM users WHERE email = ?", (request.email,))
    existing = cursor.fetchone()
    if existing:
        if existing["auth_provider"] == "google":
            raise HTTPException(
                status_code=409,
                detail="This email is registered with Google Sign-In. Please use Google login."
            )
        if existing["is_verified"]:
            raise HTTPException(
                status_code=409,
                detail="This account is already verified. Please log in to access your AI account."
            )
        # Resend OTP for unverified accounts
    else:
        # Create unverified user
        pw_hash = hash_password(request.password)
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, auth_provider, is_verified) VALUES (?, ?, ?, 'email', 0)",
            (request.name, request.email, pw_hash)
        )
        db.commit()

    # Generate and store OTP
    otp_code = f"{secrets.randbelow(900000) + 100000}"  # 6-digit code
    expires = datetime.utcnow() + timedelta(minutes=10)
    cursor.execute(
        "INSERT INTO otp_codes (email, code, expires_at, used) VALUES (?, ?, ?, 0)",
        (request.email, otp_code, expires.isoformat())
    )
    db.commit()

    # Send OTP
    send_otp_email(request.email, otp_code)

    return {
        "message": "Verification code sent. Your account is not verified yet, so AI access is blocked until OTP verification."
    }


@router.post("/verify-otp")
async def verify_otp(request: VerifyOTPRequest, db=Depends(get_db)):
    cursor = db.cursor()

    cursor.execute(
        "SELECT * FROM otp_codes WHERE email = ? AND code = ? AND used = 0 ORDER BY id DESC LIMIT 1",
        (request.email, request.code)
    )
    otp_row = cursor.fetchone()

    if not otp_row:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code.")

    # Check expiry
    expires_at = datetime.fromisoformat(otp_row["expires_at"])
    if datetime.utcnow() > expires_at:
        raise HTTPException(status_code=400, detail="Verification code has expired. Please register again.")

    # Mark OTP as used
    cursor.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (otp_row["id"],))

    # Mark user as verified
    cursor.execute("UPDATE users SET is_verified = 1 WHERE email = ?", (request.email,))
    db.commit()

    # Return user info
    cursor.execute("SELECT name, email FROM users WHERE email = ?", (request.email,))
    user = cursor.fetchone()

    return {"message": "Account verified successfully!", "user": {"name": user["name"], "email": user["email"]}}


@router.post("/login")
async def login(request: LoginRequest, db=Depends(get_db)):
    cursor = db.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (request.email,))
    user = cursor.fetchone()

    if not user:
        raise HTTPException(status_code=401, detail="No account found with this email. Please create an account first.")

    if user["auth_provider"] == "google":
        raise HTTPException(status_code=401, detail="This account uses Google Sign-In. Please use the Google button.")

    if not user["is_verified"]:
        raise HTTPException(
            status_code=403,
            detail="Account not verified. Please verify your email first; unverified accounts cannot access the AI bot."
        )

    if not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect password.")

    return {"message": "Login successful", "user": {"name": user["name"], "email": user["email"]}}


@router.get("/me")
async def get_current_user(x_user_email: EmailStr = Header(...), db=Depends(get_db)):
    """Validate current user and ensure account is verified before allowing app access."""
    cursor = db.cursor()
    cursor.execute("SELECT name, email, is_verified FROM users WHERE email = ?", (x_user_email,))
    user = cursor.fetchone()

    if not user:
        raise HTTPException(status_code=401, detail="Account not found. Please sign up with a valid account.")

    if not user["is_verified"]:
        raise HTTPException(
            status_code=403,
            detail="Your account is not verified yet. Please verify your account to access the AI bot."
        )

    return {"user": {"name": user["name"], "email": user["email"]}}


@router.post("/google")
async def google_auth(request: GoogleAuthRequest, db=Depends(get_db)):
    """Handle Google sign-in. Creates account if new, logs in if existing."""
    cursor = db.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (request.email,))
    user = cursor.fetchone()

    if not user:
        # Auto-create verified account for Google users
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, auth_provider, is_verified) VALUES (?, ?, NULL, 'google', 1)",
            (request.name, request.email)
        )
        db.commit()

    return {"message": "Google auth successful", "user": {"name": request.name, "email": request.email}}


from fastapi.responses import HTMLResponse

@router.get("/users", response_class=HTMLResponse)
async def list_users(db=Depends(get_db)):
    """Admin endpoint to see all registered users"""
    cursor = db.cursor()
    cursor.execute("SELECT id, name, email, auth_provider, is_verified, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    
    html_content = """
    <html>
        <head>
            <title>Registered Users</title>
            <style>
                body { font-family: system-ui, sans-serif; padding: 40px; background: #0f1115; color: #f8fafc; }
                h1 { color: #8b5cf6; }
                table { border-collapse: collapse; width: 100%; max-width: 1000px; margin-top: 20px; background: rgba(26,29,36,0.9); border-radius: 8px; overflow: hidden; }
                th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08); }
                th { background: rgba(139,92,246,0.1); color: #c4b5fd; text-transform: uppercase; font-size: 12px; }
                tr:hover { background: rgba(255,255,255,0.02); }
                .verified { color: #10b981; font-weight: bold; }
                .unverified { color: #ef4444; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Registered Users Database</h1>
            <table>
                <tr>
                    <th>ID</th><th>Name</th><th>Email</th><th>Auth Provider</th><th>Status</th><th>Registered At</th>
                </tr>
    """
    
    for u in users:
        status_class = "verified" if u["is_verified"] else "unverified"
        status_text = "Verified" if u["is_verified"] else "Unverified"
        html_content += f"""
            <tr>
                <td>{u["id"]}</td>
                <td>{u["name"]}</td>
                <td>{u["email"]}</td>
                <td>{u["auth_provider"]}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{u["created_at"]}</td>
            </tr>
        """
        
    html_content += """
            </table>
        </body>
    </html>
    """
    return html_content
