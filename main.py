from zigbang import *

# # get room ids
# os.makedirs('./{}'.format('zigbang'), exist_ok=True)
# ids = []
# params = OneRoomRequest('wydmc')
# response = requests.get(params.host, params = vars(params))
# response = BangResponse(response.json())
#
# for section in response.sections:
#     ids += section['item_ids']
#
# ids = list(set(ids))
# print('items num: {}'.format(len(ids)))
# req = OneRoomInfoRequest()
# '''
# url = 'http://example.com/img.png'
# response = requests.get(url, stream=True)
# with open('img.png', 'wb') as out_file:
#     shutil.copyfileobj(response.raw, out_file)
# del response
# '''
# total = 0
# for k in ids:
#     host = req.host
#     res = requests.get(host.format(k)).json()
#     for i,image in enumerate(res['item']['images']):
#         print('{}: Downloading... {}'.format(total,image))
#         # split ext https://stackoverflow.com/questions/541390/extracting-extension-from-filename-in-python
#         with open('./{}/{}_{}.{}'.format('zigbang', k,i,  os.path.splitext(image)[1][1:]), 'wb') as img:
#             shutil.copyfileobj(requests.get(image, params={'w':256}, stream=True).raw, img)
#             total += 1

#wyd7f
#wydmd
#wydm0-7

#'wyd','wye','wy6','wy7'
#0123456789bcfdgeuskhvtmjywqnzxrp
zb = ZigBangCrawler()
zb.crawling(['wydm' + i for i in '89bcfgeuskhvtmjywqnzxrp'],width=256, height=256)




