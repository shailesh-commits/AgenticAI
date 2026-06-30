import msal
import requests

# Paste your credentials from Microsoft Entra admin center

CLIENT_ID = "bbbce114-dd92-480e-96a6-8397d5f437a5"
TENANT_ID = "xoriant"  # Use "common" for personal & work accounts
AUTHORITY = f"https://login.microsoftonline.com/xoriota.onmicrosoft.com"

CLIENT_SECRET = "d798Q~0WTI6381e3bfoGWdtGmzvL~xcav4szobpf"
# Use your tenant domain or the 36-character Tenant ID GUID



# Application permissions require the '.default' scope
SCOPES = ["https://graph.microsoft.com/.default"]

def get_access_token():
    # Initialize ConfidentialClientApplication to accept the secret
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET
    )
    
    # Check MSAL cache first
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result:
            return result['access_token']
             

    # Authenticate directly using the Client Secret (Client Credentials Flow)
    result = app.acquire_token_for_client(scopes=SCOPES)

    print(result['access_token'] )

    
    if "access_token" in result:
        return result['access_token']
    else:
        raise Exception(f"Authentication failed: {result.get('error_description')}")

def list_onedrive_files(token):
    # For background/service apps, you must target a specific user's drive or a site ID
    # Replace 'user@xoriant.com' with the target user's email address
    TARGET_USER = "user@xoriota.onmicrosoft.com"
    endpoint = f"https://microsoft.com{TARGET_USER}/drive/root/children"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        items = response.json().get('value', [])
        print(f"\n--- OneDrive Root Files for {TARGET_USER} ---")
        for item in items:
            print(f"- {item['name']} ({'Folder' if 'folder' in item else 'File'})")
    else:
        print(f"Graph API Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    try:
        print("Authenticating with Client Secret...")
        access_token = get_access_token()
        print("Authentication successful!")
        
        # Test the connection
        list_onedrive_files(access_token)
    except Exception as e:
        print(f"An error occurred: {e}")
