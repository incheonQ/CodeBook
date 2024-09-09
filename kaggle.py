
def download_dataset(name-of-competition):
  """
    Downloads a dataset from Kaggle using Google Colab.

    Parameters:
    name_of_competition (str): The name of the Kaggle competition or dataset.

    Steps:
    1. Prompts the user to check if they have the kaggle.json file.
    2. Installs the Kaggle API client if not already installed.
    3. Uploads the kaggle.json file using Colab's file upload interface.
    4. Configures the Kaggle API client with the uploaded kaggle.json file.
    5. Downloads the dataset associated with the specified competition.
  """

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
