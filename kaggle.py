
def download_dataset(name-of-competition):

  ans = 'Do you download kaggle.json file on your machine?(y/n)'

  if ans == 'y':

    #Reference: https://www.kaggle.com/discussions/general/74235
    !pip install -q kaggle

    from google.colab import files
    
    print('Choose the kaggle.json file that you downloaded')
    files.upload()

    !mkdir ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    !kaggle datasets list

    !kaggle competitions download -c 'name-of-competition'

    print('Done')

  elif ans == 'n':

    print('''1. Go to your account, Scroll to API section and Click Expire API Token to remove previous tokens
             2. Click on Create New API Token - It will download kaggle.json file on your machine.''')
    
  else:
    print('Invalid input')
