# import os
# from email.message import EmailMessage
# import ssl
# import smtplib
#
# email_sender = 'huynhduy25072002@gmail.com'
# email_password = os.environ.get('EMAIL_PASSWORD')
# email_receiver='huynhgiaduy2507@gmail.com'
# subject = 'ss'
# body = ' hello world'
#
# em = EmailMessage()
# em['From'] = email_sender
# em['To'] = email_receiver
# em['Subject'] = subject
#
# em.set_content(body)
# context = ssl.SSLContext(ssl.PROTOCOL_TLS)
#
# with smtplib.SMTP_SSL('smtp.gmail.com',465,context=context) as smtp:
#     smtp.login(email_sender,email_password)
#     smtp.sendmail(email_sender,email_receiver,em.as_string())
import smtplib
from builtins import input

email = 'huynhduy25072002@gmail.com'
receiver = 'huynhgiaduy2507@gmail.com'
sub = "ss"
message="hello"
text =f"subject:{sub}\n\n{message}"
server = smtplib.SMTP('smtp.gmail.com',587)
server.starttls()
server.login(email,'aeed ivtk mttw dwmf')
server.sendmail(email,receiver,text)


