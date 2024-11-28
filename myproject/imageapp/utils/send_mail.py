import smtplib
from email.message import EmailMessage

my_mail = "maithreyinayak@gmail.com"
my_password = "gkffbnfohnlpingk"

def send_email(subject, recipient_email, template_path, context):
    sender_email = my_mail
    sender_password = my_password

    # Load and format the email template
    with open(template_path, 'r') as file:
        html_content = file.read()

    # Replace placeholders in the template
    for key, value in context.items():
        html_content = html_content.replace(f"{{{{ {key} }}}}", value)

    # Create the email
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.set_content("This is a fallback plain text message.")
    msg.add_alternative(html_content, subtype='html')  # Add the HTML content

    # Send the email using Gmail's SMTP server
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Example usage
# send_templated_email(
#     subject="Welcome to Our Service!",
#     recipient_email="recipient@example.com",
#     template_path="email_template.html",
#     context={"name": "Uttam"}
# )
