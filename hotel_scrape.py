#pip install playwright pandas openpyxl
from playwright.sync_api import sync_playwright
import pandas as pd

def main():
    
    with sync_playwright() as p:
        
        reviews_list = []
        #playwright install chromium
        browser = p.chromium.launch(headless= False)
        def getReviews(page):
            page_url = f"https://www.tripadvisor.com/Hotel_Review-g60982-d87119-Reviews-or{page}-Hotel_La_Croix-Honolulu_Oahu_Hawaii.html#REVIEWS"

            
            page = browser.new_page()
            page.goto(page_url,timeout=60000)

            reviews = page.locator("//div[@class = 'YibKl MC R2 Gi z Z BB pBbQr']").all()
                
            print(f'There are: {len(reviews)} reviews.')

            
            for review in reviews:
                review_dict = {}
                review_dict['review'] = review.locator('//div[@class="fIrGe _T"]').inner_text()
                
                reviews_list.append(review_dict)
                    
            return

        for x in range(10,70,10):
            getReviews(x)
        
        df = pd.DataFrame(reviews_list)
        df.to_excel('reviews_list1.xlsx', index=False) 
        df.to_csv('reviews_list1.csv', index=False) 

if __name__ == '__main__':
    main()