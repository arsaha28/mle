"""Run this script to regenerate data/laptop_policy.pdf.
Requires: pip install reportlab
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import os


def build_pdf(path: str) -> None:
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2.5*cm, rightMargin=2.5*cm, topMargin=2.5*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=6, alignment=TA_CENTER)
    sub_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10, textColor="#555555", alignment=TA_CENTER, spaceAfter=20)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=4)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=16, alignment=TA_JUSTIFY, spaceAfter=8)
    bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=10, leading=15, leftIndent=20, spaceAfter=4)

    story = [
        Paragraph("Acme Corporation", title_style),
        Paragraph("Employee Laptop & Device Usage Policy", title_style),
        Paragraph("Version 2.1 | Effective Date: January 1, 2025 | HR & IT Department", sub_style),
        HRFlowable(width="100%", thickness=1, color="#cccccc"),
        Spacer(1, 14),
    ]

    sections = [
        ("1. Purpose", ["This policy establishes guidelines for acceptable use of company-issued laptops and other computing devices. The purpose is to protect the confidentiality, integrity, and availability of company information assets."], []),
        ("2. Scope", ["Applies to all employees, contractors, consultants, and any third party with access to a company-owned or company-managed computing device."], []),
        ("3. Acceptable Use", ["Company laptops are provided primarily for business purposes. Limited personal use is permitted provided it does not interfere with job responsibilities. Acceptable use includes:"], ["Conducting work-related tasks, communications, and research.", "Accessing company-approved software, cloud services, and internal systems.", "Attending virtual meetings via approved platforms.", "Limited personal browsing during breaks.", "Installing software approved by the IT department."]),
        ("4. Prohibited Activities", ["The following activities are strictly prohibited:"], ["Accessing, downloading, or distributing illegal or inappropriate content.", "Installing unlicensed software or circumventing licensing agreements.", "Disabling security software, firewalls, or endpoint protection.", "Using devices for personal commercial activities.", "Sharing login credentials or allowing unauthorised individuals to use your device.", "Connecting to public Wi-Fi without the company-approved VPN.", "Mining cryptocurrency or running unauthorised background processes.", "Attempting to gain unauthorised access to any network or data."]),
        ("5. Security Requirements", ["All employees must adhere to the following security requirements:"], ["Lock your screen when leaving your workstation unattended.", "Ensure full-disk encryption is enabled (BitLocker/FileVault).", "Keep the OS and all applications updated as prompted by IT.", "Use only company-approved cloud storage for work files.", "Never store sensitive data on personal storage devices.", "Report lost or stolen devices to IT Security within one hour.", "Use multi-factor authentication (MFA) on all company accounts."]),
        ("6. Password Policy", ["Strong password hygiene is critical. Employees must:"], ["Use passwords of at least 12 characters with uppercase, lowercase, numbers, and symbols.", "Never reuse passwords across different systems.", "Use the company-approved password manager (1Password).", "Change passwords immediately if compromise is suspected.", "Never write passwords down or share them via email or chat."]),
        ("7. Remote Work & Travel", ["Employees working remotely or travelling must:"], ["Always use the company VPN when accessing internal systems remotely.", "Avoid working on sensitive information in public spaces.", "Use a privacy screen filter in public locations.", "Never leave devices unattended in vehicles or public areas.", "Notify IT if the device crosses international borders."]),
        ("8. Software & Application Management", ["The IT department manages all software on company devices. Employees must:"], ["Request new software through the IT helpdesk or self-service portal.", "Allow automatic updates to run and reboot devices promptly.", "Refrain from using browser extensions not approved by IT.", "Notify IT if software is behaving unexpectedly."]),
        ("9. Data Handling & Confidentiality", ["All employees must handle data per the company Data Classification Policy:"], ["Confidential data must be encrypted in transit and at rest.", "Do not email confidential data to personal email accounts.", "Dispose of sensitive documents using secure shredding bins.", "Customer data must only be processed in designated approved systems."]),
        ("10. Monitoring & Privacy", ["Company-owned devices are subject to monitoring. Acme Corporation reserves the right to:"], ["Monitor network traffic, application usage, and web browsing on company devices.", "Remotely access, lock, or wipe lost, stolen, or compromised devices.", "Audit installed software and configurations for compliance.", "Review device logs during security incident investigations."]),
        ("11. Device Return & Offboarding", ["Upon departure or role change, employees must:"], ["Return all company devices to IT within their last working day.", "Ensure no personal data remains on the device prior to return.", "Return all accessories including chargers and peripherals.", "Not attempt to factory-reset devices independently."]),
        ("12. Consequences of Policy Violations", ["Violations may result in disciplinary action up to and including termination, as well as potential civil or criminal liability. Suspected violations should be reported to IT Security or HR immediately."], []),
        ("13. Policy Review", ["This policy is reviewed annually by IT Security and HR, or more frequently in response to significant changes in technology or legislation."], []),
        ("14. Contact", ["For questions regarding this policy, please contact:"], ["IT Helpdesk: itsupport@acmecorp.example.com | ext. 1100", "IT Security: security@acmecorp.example.com | ext. 1200", "HR Department: hr@acmecorp.example.com | ext. 1300"]),
    ]

    for heading, paragraphs, bullets in sections:
        story.append(Paragraph(heading, h2_style))
        for p in paragraphs:
            story.append(Paragraph(p, body_style))
        for b in bullets:
            story.append(Paragraph(f"• {b}", bullet_style))
        story.append(Spacer(1, 6))

    story.extend([
        HRFlowable(width="100%", thickness=1, color="#cccccc"),
        Spacer(1, 8),
        Paragraph("By using a company-issued device, you acknowledge that you have read, understood, and agree to comply with this policy.",
                  ParagraphStyle("Footer", parent=styles["Italic"], fontSize=9, alignment=TA_CENTER, textColor="#777777")),
    ])

    doc.build(story)
    print(f"PDF written to {path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "laptop_policy.pdf")
    build_pdf(out)
