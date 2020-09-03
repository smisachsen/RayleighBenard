import os


def create_folder_if_not_exists(folderpath):
    if os.path.exists(folderpath):
        return

    os.mkdir(folderpath)
    return

def get_new_subfolder(folderpath):
    create_folder_if_not_exists(folderpath)

    current_folders = os.listdir(folderpath)
    new_folder_name = f"{len(current_folders)}/"

    new_path = folderpath + new_folder_name
    create_folder_if_not_exists(new_path)

    return new_path

if __name__ == '__main__':
    get_new_subfolder("test_models/")
