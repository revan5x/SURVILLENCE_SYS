import json
import smtplib
from email.mime.text import MIMEText

# Load config
with open('email_config.json', 'r') as f:
    config = json.load(f)

try:
    server = smtplib.SMTP(config['smtp_server'], config['smtp_port'], timeout=10)
    server.starttls()
    server.login(config['sender_email'], config['sender_password'])
    
    msg = MIMEText("Test alert from Surveillance System")
    msg['Subject'] = 'Test Alert'
    msg['From'] = config['sender_email']
    msg['To'] = ', '.join(config['recipient_emails'])
    
    server.sendmail(config['sender_email'], config['recipient_emails'], msg.as_string())
    server.quit()
    print("✓ Email test successful!")
except Exception as e:
    print(f"✗ Email test failed: {e}")