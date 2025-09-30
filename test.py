import requests

url = "https://storage.googleapis.com/kagglesdsdata/datasets/601280/1079953/lung_colon_image_set/lung_image_sets/lung_aca/lungaca10.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250927%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250927T171854Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=73c38081060ecec10ccc02d6d25f0529591912a97c3b1c6f543bc72393530ff98add04932c63eda81821afa0882989a953a9a7c80fc0c368a328f20efc129a98408e92b3ae6f0da0bb5b443ed60524c4df02cd984bf6c18b687f59d92cb6ad70d146015e0ae6834b564555fc412a451861199b352354ac0fd82b70b15806dd3f2ff11480ff37af090bf23431d4b2afc5eec7eecca54898e5b29b8083d32b08916be0824b774ac9790f26b777d7f0973cd13606ef7468723dfd23faad0d5e6c25256a32bf40054404f6d2088c11a053f316dc60a89ef5b62c97d0024ed5342cb05a44b0ac362c6a5b056b67d34e9ee1931e87065052d4c8fd6731f5cd4860f700"

r = requests.get(url, stream=True)
with open("lungaca10.jpeg", "wb") as f:
    for chunk in r.iter_content(1024):
        f.write(chunk)

print("Download complete!")
