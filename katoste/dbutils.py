import requests

def get_from_geneid(geneid, feature='cdna'):
    server = "https://rest.ensembl.org"
    ext = f"/xrefs/symbol/homo_sapiens/{geneid}?content-type=text/plain"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
    decoded = r.json()

    ext = f"/sequence/id/{decoded[0]['id']}?content-type=text/plain;type={feature};species=homo_sapiens"
    r = requests.get(server+ext, headers={ "Content-Type" : "text/plain"})
    if not r.ok:
        r.raise_for_status()
    
    return r.text


def get_from_ensemblid(geneid, feature='cdna'):
    import string
    table = str.maketrans('', '', string.ascii_lowercase)

    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{geneid}?content-type=text/plain;type={feature};species=homo_sapiens;mask_feature=1"
    r = requests.get(server+ext, headers={ "Content-Type" : "text/plain"})
    if not r.ok:
        r.raise_for_status()
    
    return r.text.translate(table)