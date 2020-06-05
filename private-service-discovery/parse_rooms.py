# room finder: https://campus.tum.de/tumonline/webnav.ini
# building: 5605 FMI / building part 5, boltzmannstr. 3
from bs4 import BeautifulSoup

html_doc = open("./rooms.html", "r").read()
soup = BeautifulSoup(html_doc, "html.parser")
table = soup.find("table", {"class": "list"})

header = table.find("thead")
header = header.find_all("th")
header = [col.text.strip() for col in header]

table_data = list()
table_body = table.find("tbody")
rows = table_body.find_all("tr")
for row in rows:
    cols = row.find_all("td")
    cols = [col.text.strip() for col in cols]
    table_data.append(cols)

print(header)
print(len(table_data))
print(table_data)
