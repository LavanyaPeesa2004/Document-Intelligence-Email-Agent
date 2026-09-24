from email.message import EmailMessage
import os
import smtplib

from dotenv import load_dotenv
from langchain_core.tools import tool


load_dotenv()


# ==================================================
# EMAIL CONFIGURATION
# ==================================================

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")


# ==================================================
# EMAIL SENDING TOOL
# ==================================================

@tool
def send_email(
    recipient: str,
    subject: str,
    body: str
) -> str:
    """
    Send an email using the configured Gmail account.

    Use this tool only after the user has explicitly
    approved the email draft.
    """

    if not EMAIL_ADDRESS:
        return "Email sending failed: EMAIL_ADDRESS is not configured."

    if not EMAIL_APP_PASSWORD:
        return "Email sending failed: EMAIL_APP_PASSWORD is not configured."

    try:

        message = EmailMessage()

        message["From"] = f"{os.getenv('SENDER_NAME', 'User')} <{EMAIL_ADDRESS}>"
        message["To"] = recipient
        message["Subject"] = subject

        message.set_content(body)

        with smtplib.SMTP_SSL(
            SMTP_SERVER,
            SMTP_PORT
        ) as server:

            server.login(
                EMAIL_ADDRESS,
                EMAIL_APP_PASSWORD
            )

            server.send_message(message)

        return f"Email successfully sent to {recipient}."

    except Exception as e:

        return f"Email sending failed: {e}"