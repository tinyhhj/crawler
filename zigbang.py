import argparse
import os
import requests
import shutil
class BangResponse:
    def __init__(self,response):
        # self.clusters = response['clusters']
        self.sections = response['sections']
'''
# 빌라 매매
https://apis.zigbang.com/v2/items?domain=zigbang&geohash=wyd5t&new_villa=true&sales_type_in=%EB%A7%A4%EB%A7%A4&zoom=13
domain: zigbang
geohash: wydhy
new_villa: true
sales_type_in: 매매
zoom: 13
# 빌라 전월세
https://apis.zigbang.com/v2/items?domain=zigbang&geohash=wyd5g&new_villa=true&sales_type_in=%EC%A0%84%EC%84%B8%7C%EC%9B%94%EC%84%B8&zoom=13
domain: zigbang
geohash: wydk9
new_villa: true
sales_type_in: 전세|월세
zoom: 13
# 원룸 전체 
https://apis.zigbang.com/v2/items?deposit_gteq=0&domain=zigbang&geohash=wydm4&rent_gteq=0&sales_type_in=%EC%A0%84%EC%84%B8%7C%EC%9B%94%EC%84%B8&service_type_eq=%EC%9B%90%EB%A3%B8
deposit_gteq: 0
domain: zigbang
geohash: wydm4
rent_gteq: 0
sales_type_in: 전세|월세
service_type_eq: 원룸
# 오피스텔 전체
https://apis.zigbang.com/v2/officetels?buildings=true&domain=zigbang&geohash=wyd79
buildings: true
domain: zigbang
geohash: wyd79
'''
class ZigbangRequest:
    def __init__(self, geohash):
        self.domain = 'zigbang'
        self.host = 'https://apis.zigbang.com/v2/items'
        self.geohash = geohash

class VillaRequest(ZigbangRequest):
    def __init__(self,geohash,sales_type_in='전세|월세',zoom=15):
        super(VillaRequest, self).__init__(geohash)
        self.new_villa = 'true'
        self.sales_type_in = sales_type_in
        self.zoom = zoom
class OneRoomRequest(ZigbangRequest):
    def __init__(self, geohash, sales_type_in='전세|월세', deposit_gteq=0, rent_gteq = 0):
        super(OneRoomRequest, self).__init__(geohash)
        self.service_type_eq = '원룸'
        self.sales_type_in = sales_type_in
        self.deposit_gteq = deposit_gteq
        self.rent_gteq=rent_gteq
class OfficetelRequest(ZigbangRequest):
    def __init__(self,geohash):
        super(OfficetelRequest, self).__init__(geohash)
        self.buildings = 'true'
        self.host = 'https://apis.zigbang.com/v2/officetels'
class OneRoomInfoRequest(ZigbangRequest):
    def __init__(self):
        super(OneRoomInfoRequest, self).__init__('')
        self.host = 'https://apis.zigbang.com/v2/items/{}'
class ZigBangCrawler:
    def __init__(self):
        os.makedirs('./{}'.format('zigbang'), exist_ok=True)

    def crawling(self, geohashes,width = 512, height=512):
        item_host = OneRoomInfoRequest().host
        total = 0
        # 5level hashes
        if geohashes is None:
            hashes = '0123456789bcfdgeuskhvtmjywqnzxrp'
            geohashes = []
            for hash in ['wyd','wye','wy6','wy7']:
                for h1 in hashes:
                    for h2 in hashes:
                        geohashes.append(hash+h1+h2)
        else:
            new_geohashes = []
            hashes = '0123456789bcfdgeuskhvtmjywqnzxrp'
            for hash in geohashes:
                if len(hash) ==5:
                    new_geohashes.append(hash)
                    continue
                for h1 in hashes:
                    if len(hash+h1) ==5:
                        new_geohashes.append(hash+h1)
                        continue
                    for h2 in hashes:
                        new_geohashes.append(hash+h1+h2)
            geohashes = new_geohashes

        print('searching hashes {} {}'.format(len(geohashes), geohashes))
        # collect ids
        ids = []
        for hash in geohashes:
            arr = [OneRoomRequest(hash),
                    VillaRequest(hash),
                    VillaRequest(hash, '매매'),
                    OfficetelRequest(hash)]

            for params in arr:
                response = requests.get(params.host, params=vars(params))
                if response.status_code != 200:
                    print('response status is bad {}/{}/{}'.format(params.host,vars(params),response.status_code))
                    continue
                response = BangResponse(response.json())

                for section in response.sections:
                    ids += section['item_ids']

                ids = list(set(ids))
        print('items num: {}'.format(len(ids)))
        # collect imgs
        for id in ids:
            try:
                itemInfo = requests.get(item_host.format(id)).json()
                for i, image in enumerate(itemInfo['item']['images']):
                    # file exists skip
                    filename = './{}/{}_{}.{}'.format('zigbang', id, i, os.path.splitext(image)[1][1:])
                    if os.path.exists(filename):
                        print('{} is exists skip downloading..'.format(filename))
                        continue
                    print('{}: Downloading... {}'.format(total, image))
                    with open(filename,'wb') as img:
                        shutil.copyfileobj(requests.get(image, params={'w': width, 'h': height}, stream=True).raw, img)
                total += 1
            except Exception as e:
                print('error occur {} moving next id'.format(id))
                print(e)
                continue

