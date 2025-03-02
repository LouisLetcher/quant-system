import os
from jinja2 import Environment, FileSystemLoader
import pandas as pd

class ReportGenerator:
    """Generates HTML reports using Jinja templates."""

    TEMPLATE_DIR = "src/reports/templates"

    def __init__(self):
        self.env = Environment(loader=FileSystemLoader(self.TEMPLATE_DIR))

    def generate_report(self, data: dict, template_name: str, output_path: str):
        """Generates an HTML report from a template and data."""
        template = self.env.get_template(template_name)
        rendered_html = template.render(data)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(rendered_html)

        return output_path