import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
from urllib.parse import urljoin, urlparse
import hashlib
import time

# Create directory for storing images only
os.makedirs('scraped_data/images', exist_ok=True)

class CriminalDataCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.session = requests.Session()
        self.downloaded_images = set()
    
    def download_image(self, img_url, base_url, record_id):
        """Download image from URL and save locally"""
        try:
            full_url = urljoin(base_url, img_url)
            
            img_hash = hashlib.md5(full_url.encode()).hexdigest()
            if img_hash in self.downloaded_images:
                return None
            
            response = self.session.get(full_url, headers=self.headers, timeout=15, stream=True)
            response.raise_for_status()
            
            ext = os.path.splitext(urlparse(full_url).path)[1]
            if not ext or ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                content_type = response.headers.get('content-type', '')
                ext = '.jpg' if 'jpeg' in content_type else '.png'
            
            filename = f"criminal_{record_id}_{img_hash}{ext}"
            filepath = os.path.join('scraped_data', 'images', filename)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.downloaded_images.add(img_hash)
            print(f"✓ Downloaded image: {filename}")
            return filepath
            
        except Exception as e:
            print(f"✗ Error downloading image {img_url}: {str(e)}")
            return None
    
    def scrape_images(self, urls):
        """Scrape and download images from provided URLs"""
        total_images = 0
        
        for url in urls:
            try:
                print(f"\n{'='*60}")
                print(f"Scraping images from: {url}")
                print(f"{'='*60}")
                
                response = self.session.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all images on the page
                img_tags = soup.find_all('img')
                print(f"Found {len(img_tags)} image tags")
                
                downloaded_count = 0
                for idx, img in enumerate(img_tags):
                    img_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    
                    # Skip icons, logos, banners
                    if img_url and not any(x in img_url.lower() for x in ['icon', 'logo', 'banner', 'button']):
                        result = self.download_image(img_url, url, idx)
                        if result:
                            downloaded_count += 1
                
                total_images += downloaded_count
                print(f"✓ Downloaded {downloaded_count} images from this page")
                
                time.sleep(2)
                    
            except Exception as e:
                print(f"✗ Error scraping {url}: {str(e)}")
        
        print(f"\n{'='*60}")
        print(f"Total images downloaded: {total_images}")
        print(f"{'='*60}\n")
        return total_images

# Example usage
if __name__ == '__main__':
    crawler = CriminalDataCrawler()
    
    # Add your URLs here
    urls_to_scrape = [
        'https://mugshots.com/',
    ]
    
    print("\n" + "="*60)
    print("Criminal Data Image Downloader")
    print("="*60)
    print("\nStarting image download...")
    print("="*60 + "\n")
    
    # Run the scraper
    crawler.scrape_images(urls_to_scrape)
