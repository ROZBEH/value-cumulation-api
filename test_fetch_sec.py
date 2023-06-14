import os
import time
import datetime
from sec_api import QueryApi, RenderApi
from dotenv import load_dotenv

load_dotenv()


def fetch_sec_report(report_type="10-K", ticker="AAPL", year=None, num_years=10):
    if year is None:
        year = (
            datetime.datetime.now().year - 2
            if datetime.datetime.now().month == 1
            else datetime.datetime.now().year - 1
        )
    SEC_API_KEY = os.getenv("SEC_API_KEY")
    queryApi = QueryApi(api_key=SEC_API_KEY)
    renderApi = RenderApi(api_key=SEC_API_KEY)

    query = {
        "query": {
            "query_string": {
                "query": f'ticker:{ticker} AND filedAt:{{{year-num_years}-11-30 TO {year+1}-11-30}} AND formType:"{report_type}"'
            }
        },
        "from": "0",
        # "size": "10", # 10 of these documents
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    filings = queryApi.get_filings(query)
    sec_filings = []
    # print(len(filings["filings"]))
    substring = "/ix?doc="
    for filing in filings["filings"]:
        sec_url = filing["linkToFilingDetails"]
        if substring in sec_url:
            sec_url = sec_url.replace(substring, "")
        filing = renderApi.get_filing(sec_url)
        # sec_filings.append(filing)
    # html_file = os.path.join(current_dir, "tmp_" + ticker + ".html")
    # with open(html_file, "w") as f:
    #     f.write(filing)

    return sec_filings


print(fetch_sec_report())
