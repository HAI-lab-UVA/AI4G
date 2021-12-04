from google.oauth2 import service_account
from googleapiclient.discovery import MEDIA_BODY_PARAMETER_DEFAULT_VALUE, build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import json
import io
import os

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = os.path.join(os.getcwd(), 'Service_Account_Key', 'ai4good-forall-925493922537.json')
FEEDBACK_RESPONSES_PATH = os.path.join(os.getcwd(), 'Feedback_Responses')
DIRECTORY_ID = '1KIvuQjfrq1lu95WfCSjDvJD6rLGOU7Yc'

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=credentials)

def invalid():
    print('Invalid option.')

# list all accessible files
def list_files():
    results = drive_service.files().list(pageSize=10).execute()
    print(json.dumps(results, indent=4))

# upload .xlsx to 'Feedback Survey' Google Drive folder
def upload():
    selected_day = input("Enter the feedback day to upload: ")
    file_name = None
    for file in os.listdir(FEEDBACK_RESPONSES_PATH):
        if selected_day in file:
            file_name = file
            break
    
    file_metadata = {
        'name': file_name,
        'parents': [DIRECTORY_ID],
        'mimeType': 'application/vnd.google-apps.spreadsheet'
    }

    media = MediaFileUpload(os.path.join(FEEDBACK_RESPONSES_PATH, file_name), resumable=True)
    file = drive_service.files().create(body=file_metadata, media_body=media).execute()
    print("File successfully uploaded!")
    print(json.dumps(file, indent=4))

# update .xlsx in Google Drive Folder
'''def update():
    media = MediaFileUpload(FEEDBACK_RESPONSES_PATH, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    file = drive_service.files().update(fileId=FILE_ID, media_body=media).execute()
    print('File successfully updated!')
    print(json.dumps(file, indent=4))'''

# download .xlsx from Google Drive
def download():
    selected_day = input("Enter the feedback day to download the spreadsheet: ")

    # search for file
    file_id = None
    file_name = None
    results = drive_service.files().list(pageSize=10, fields='files(id,name,mimeType)').execute()
    for file in results['files']:
        if selected_day in file['name'] and file['mimeType'] == 'application/vnd.google-apps.spreadsheet':
            file_id = file['id']
            file_name = "{}.csv".format(file['name'])

    if file_id == None:
        print("No file found.")
        exit()
    
    request = drive_service.files().export_media(fileId=file_id, mimeType='text/csv')

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    with open(os.path.join(FEEDBACK_RESPONSES_PATH, file_name), 'wb') as f:
        f.write(fh.getbuffer())
    
    print('File successfully downloaded!')
    
if __name__ == '__main__':
    selected_option = input("Type 'upload' to upload generated feedback file or 'download' to download the file: ")
    switch = {
        # 'list': list_files,
        'upload': upload,
        # 'update' : update, 
        'download': download
    }
    func = switch.get(selected_option, invalid)
    func()