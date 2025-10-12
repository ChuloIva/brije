#!/usr/bin/env python3
import subprocess
import sys
import os

def check_dependencies():
    """Check if required tools are installed."""
    tools = {
        'pandoc': 'brew install pandoc',
        'pdflatex': 'brew install --cask mactex-no-gui'
    }

    missing = []
    for tool, install_cmd in tools.items():
        result = subprocess.run(['which', tool], capture_output=True, text=True)
        if result.returncode != 0:
            missing.append((tool, install_cmd))

    return missing

def convert_with_pandoc(md_path, pdf_path):
    """Convert markdown to PDF using pandoc with pdflatex."""
    print(f"Converting {md_path} to PDF using pandoc...")

    cmd = [
        'pandoc',
        md_path,
        '-o', pdf_path,
        '--pdf-engine=pdflatex',
        '-V', 'geometry:margin=0.75in',
        '-V', 'fontsize=10pt',
        '-V', 'linestretch=1.2',
        '--toc',
        '--toc-depth=2',
        '-V', 'colorlinks=true',
        '-V', 'linkcolor=blue',
        '-V', 'urlcolor=blue'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ PDF created successfully: {pdf_path}")
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
        return True
    else:
        print(f"❌ Error with pdflatex:")
        print(result.stderr)
        return False

def convert_with_weasyprint(md_path, pdf_path):
    """Convert markdown to PDF using markdown2 + weasyprint."""
    try:
        import markdown2
        from weasyprint import HTML, CSS

        print(f"Converting {md_path} to PDF using weasyprint...")

        # Read markdown file
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Convert markdown to HTML
        html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks'])

        # Add CSS styling
        css = CSS(string='''
            @page {
                size: letter;
                margin: 0.75in;
            }
            body {
                font-family: "Helvetica", "Arial", sans-serif;
                font-size: 10pt;
                line-height: 1.4;
            }
            h1 {
                font-size: 18pt;
                margin-top: 20pt;
                margin-bottom: 10pt;
                color: #2c3e50;
            }
            h2 {
                font-size: 14pt;
                margin-top: 16pt;
                margin-bottom: 8pt;
                color: #34495e;
            }
            p {
                margin-bottom: 8pt;
            }
            em {
                color: #7f8c8d;
                font-style: italic;
            }
            strong {
                font-weight: bold;
                color: #2c3e50;
            }
            hr {
                border: none;
                border-top: 1px solid #bdc3c7;
                margin: 10pt 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 10pt 0;
                font-size: 9pt;
            }
            th, td {
                border: 1px solid #bdc3c7;
                padding: 4pt 8pt;
                text-align: left;
            }
            th {
                background-color: #ecf0f1;
                font-weight: bold;
            }
        ''')

        # Create HTML document
        html_doc = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Carl Rogers Therapy Sessions - Annotated Transcripts</title>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        '''

        # Convert to PDF
        HTML(string=html_doc).write_pdf(pdf_path, stylesheets=[css])

        print(f"✓ PDF created successfully: {pdf_path}")
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
        return True

    except ImportError as e:
        print(f"❌ Missing Python packages: {e}")
        print("   Install with: pip3 install markdown2 weasyprint")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    md_path = "/Users/ivanculo/Desktop/Projects/ment_helth/brije/output/carl_rogers_analysis/detailed_transcript.md"
    pdf_path = "/Users/ivanculo/Desktop/Projects/ment_helth/brije/output/carl_rogers_analysis/detailed_transcript.pdf"

    print("=" * 60)
    print("PDF Conversion Tool")
    print("=" * 60)
    print()

    # Check if markdown file exists
    if not os.path.exists(md_path):
        print(f"❌ Markdown file not found: {md_path}")
        sys.exit(1)

    # Try pandoc first
    missing = check_dependencies()

    if not missing:
        print("✓ All dependencies found (pandoc, pdflatex)")
        print()
        success = convert_with_pandoc(md_path, pdf_path)
        if success:
            return
    else:
        print("⚠️  Missing dependencies:")
        for tool, install_cmd in missing:
            print(f"    {tool}: {install_cmd}")
        print()

    # Try weasyprint as fallback
    print("Trying alternative method (weasyprint)...")
    print()
    success = convert_with_weasyprint(md_path, pdf_path)

    if not success:
        print()
        print("=" * 60)
        print("❌ PDF conversion failed with all methods")
        print("=" * 60)
        print()
        print("Alternative solutions:")
        print("  1. Install pandoc and pdflatex:")
        print("     brew install pandoc")
        print("     brew install --cask mactex-no-gui")
        print()
        print("  2. Install Python packages:")
        print("     pip3 install markdown2 weasyprint")
        print()
        print("  3. Use online converter:")
        print("     https://www.markdowntopdf.com/")
        print()
        print("  4. Open in VS Code and use Markdown PDF extension")
        sys.exit(1)

if __name__ == "__main__":
    main()