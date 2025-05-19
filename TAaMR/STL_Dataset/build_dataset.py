import json
import os
import requests

dataset_name = 'fashion'
json_file_path = f'{dataset_name}.json'
# Function to convert product/scene ID to URL
def convert_to_url(signature):
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)


# Load JSON data
data = []
with open(json_file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))
save_path = fr'../data/{dataset_name}/original_images'
# Create directories to store images if they don't exist
os.makedirs(save_path, exist_ok=True)

# Define request headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
    "Referer": "https://www.example.com",
    "Cookie": "HSID=AizYaCZUxk6VZQmw6; SSID=AfM0WYfhmfs8DHDIB; APISID=mzWIhCGWNCG6-5lR/A5MpEwmvi2z9r9Cr6; SAPISID=4TdfIzdzh36Ste9w/AB15hgsevcEKUssXT; __Secure-1PAPISID=4TdfIzdzh36Ste9w/AB15hgsevcEKUssXT; __Secure-3PAPISID=4TdfIzdzh36Ste9w/AB15hgsevcEKUssXT; receive-cookie-deprecation=1; SID=g.a000pAhpu08YJ53dweW9p31QysWno-fWQthFWB3cK4CfSZenNZP93sdxpjAaRFFF6IYjcwJU7gACgYKAYISARMSFQHGX2Mir_IG_GdN8UF8HU-nVbzR0RoVAUF8yKrtjZJzVKoaeScw9xfYXbPp0076; __Secure-1PSID=g.a000pAhpu08YJ53dweW9p31QysWno-fWQthFWB3cK4CfSZenNZP9ee_IlxbBcLtpsEpgQ807fQACgYKAbESARMSFQHGX2MiYHp-XpxIaZ7fVlt-iI2q3BoVAUF8yKpCuJ0nlAfak8m4UL_Mp7OQ0076; __Secure-3PSID=g.a000pAhpu08YJ53dweW9p31QysWno-fWQthFWB3cK4CfSZenNZP966M_cinC2wcCrv3S0jHq7wACgYKARYSARMSFQHGX2MiqE8tnju4vUK76_dt0QWRZBoVAUF8yKoeyWVOl9xQa9ZixrwQY5XS0076; AEC=AVYB7cqs3EbHRXhywzEr0r6LSp70Z0nWBumrsLh-JBpH9FovgMDuULEdC_c; SEARCH_SAMESITE=CgQIvZwB; NID=518=j3TJ5l9oVdbhlWkzDrcrCxFsWjSTRdF1UWuKglg88FQtsKiqUiRhmmNIoKquO-UqY-RKCH0aQBsC55eBzEBWlGf5Z3hunfOrvn3IM-TjhynHt-j1TXV5SqbAggXYjPb5RW8CV8PSGX_UFhJuAbiRug5kEYthNLKwG8juAleMHkdSO__pU4B_56apIY0AzMI2mObrByZSwDbL51Vx1CxGeICZ247fhpaXNfRtJ8ErO-rvq2FQ9WtA_XZMOLOdPfNkQP6LgN7GARd2dCk9dtFaJSkkGl3MgsBc4YFc3khcpuVNtvulYdV87hoCXVSizmLu8dlB58qtfFL556unuVSPfcuLH_eOD58ovS42Yx_peaUj9NVspof5ZCVb1eTlnarvaPNtA_tC7C540I-2OWDIMI9hl1U-kHIzQM49KZEWvDwjDSmo6-lTbUfyXR4rflokDY5HYYlShsJXM4zuwM5UnRJR; __Secure-1PSIDTS=sidts-CjIBQT4rX9t_REryZBc-TITmNrDa4ZNrmvh8farSjkkUvUyxWkIiIjQXNkqRxhI11cQ6fBAA; __Secure-3PSIDTS=sidts-CjIBQT4rX9t_REryZBc-TITmNrDa4ZNrmvh8farSjkkUvUyxWkIiIjQXNkqRxhI11cQ6fBAA; DV=Q0MyT_tVUlpUEL5WPJip-J-GsRvdLpl7gAFAyE5G1AAAAFBxZ8UCkzF_bwAAAKBhEoxfPqk8PAAAAAHiplD9Std3EAAAAA; SIDCC=AKEyXzXxLEcOV1Tc7iXaMR614ZiL7E3NszXP2k5qB1wct5dbrXYUwCiS1aq1AZx97R300HBBgdU; __Secure-1PSIDCC=AKEyXzXZNBtEeaNl7Cq-s9G8xkYwQCdFtLbtJ7kaga5S1GBdzDc-2UBNNwwLkDg_15ptmO9WXfk; __Secure-3PSIDCC=AKEyXzW_F6I70gtHrSwIn18KfH8K_o_DNKsSljCTaQouNtFboMxaVVDg5mWUp8LK14Xta6cKoAg"
}

# Initialize a dictionary to map product_id to user_id
product_to_user = {}
next_user_id = 0

# Initialize a list for interaction records
interaction_records = []
image_save_path = os.path.join(save_path, "images")
os.makedirs(image_save_path, exist_ok=True)
# Loop through each entry in the JSON file
for idx, entry in enumerate(data):
    product_id = entry['product']
    scene_id = entry['scene']
    bbox = entry['bbox']

    # Map product_id to a user_id if not already mapped
    if product_id not in product_to_user:
        product_to_user[product_id] = next_user_id
        next_user_id += 1

    # Get user_id from mapping
    user_id = product_to_user[product_id]

    # Get URL for scene image
    url = convert_to_url(scene_id)

    # Download and save scene image with headers
    img_path = os.path.join(image_save_path, f"{idx}.jpg")  # Image saved as 0.jpg, 1.jpg, etc.
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(img_path, 'wb') as img_file:
            img_file.write(response.content)
        print(f"Downloaded {scene_id} as {img_path}")
    else:
        print(f"Failed to download {scene_id}. Status code: {response.status_code}")

    # Record interaction as "user_id[null]scene_index"
    interaction_records.append(f"{user_id}[null]{idx}")  # product's mapped user_id as user, image index as item

interaction_file_path = os.path.join(save_path, 'pos.txt')
with open(interaction_file_path, 'w') as f:
    for record in interaction_records:
        f.write(record + '\n')
print(f"Interaction records saved to {interaction_file_path}")
print(f"Total Users:{len(next_user_id)}; Total Interactions:{idx+1}")
