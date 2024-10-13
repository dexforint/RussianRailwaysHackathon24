from bs4 import BeautifulSoup
import re


def get_title(soup: BeautifulSoup) -> str:
    title = soup.find("h2", {"class": "card-header"}).text
    title = re.sub(r"\s+", " ", title)
    return title


def get_dates(soup: BeautifulSoup) -> list:
    date_elements = soup.find_all("div", {"class": "date-top"})

    dates = []
    for el in date_elements:
        date_type = el.find("span", {"class": "text-more-grey"}).text.strip()
        date = el.text.replace(date_type, "").strip()
        dates.append({"type": date_type, "date": date})

    return dates


def get_doc_type(soup: BeautifulSoup) -> tuple[str, str]:
    doc_type = soup.find_all("div", {"class": "doc-type"})[0].text
    doc_type = doc_type.replace("\xa0", " ").strip()

    doc_type_date = re.search(r"(?<=от )\d\d\.\d\d\.\d\d\d\d", doc_type)
    doc_type_date = None if doc_type_date is None else doc_type_date.group()

    return (doc_type, doc_type_date)


def get_rubrics(soup: BeautifulSoup) -> list[str]:
    rubrics_a = soup.select(".rubrics > a")
    rubrics = [r.text.strip() for r in rubrics_a]
    return rubrics
