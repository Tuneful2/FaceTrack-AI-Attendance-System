import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "attendance_log.csv")


def generate_html_table_rows():
    """
    Reads attendance_log.csv and converts rows into
    HTML <tr> table rows for attendance.html
    """

    if not os.path.exists(CSV_PATH):
        print("[ERROR] attendance_log.csv not found!")
        return ""

    html_rows = ""

    with open(CSV_PATH, mode="r") as file:
        reader = csv.reader(file)

        # Skip header
        next(reader)

        for row in reader:
            name, date, time = row
            html_rows += f"<tr><td>{name}</td><td>{date}</td><td>{time}</td></tr>\n"

    return html_rows


if __name__ == "__main__":
    html_output = generate_html_table_rows()

    print("\n========== GENERATED HTML TABLE ROWS ==========\n")
    print(html_output)
