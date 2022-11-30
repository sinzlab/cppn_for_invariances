import re

extract_data_key = lambda path: "-".join((re.findall(r"\d+", path)[:3] + ["0"]))
