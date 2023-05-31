import shutil
import os
import torch

# def merge_pt_files(input_files, output_file):
#     with open(output_file, 'wb') as output:
#         for input_file in input_files:
#             print(f"Opening file {input_file}...")
#             with open(input_file, 'rb') as input:
                
#                 if os.path.getsize(input_file) > 0:
#                     shutil.copyfileobj(input, output)
#                     print(f"Copied file {input_file}...")


def merge_pt_files(input_files, output_file):
    merged_data = {}
    for input_file in input_files:
        print(f"Opening file {input_file}...")
        if os.path.getsize(input_file) > 0:
            checkpoint = torch.load(input_file, map_location=torch.device('cpu'))
            merged_data.update(checkpoint)
            print(f"Merged file {input_file}...")

    print(f"Saving file {output_file}...")
    torch.save(merged_data, output_file)


input_files = data_files = [
    "/vol/bitbucket/nm219/data/close_box_semimasked_1.pt",
    "/vol/bitbucket/nm219/data/close_box_semimasked_2.pt",
    # "/vol/bitbucket/nm219/data/close_box_semimasked_3.pt",
    # "/vol/bitbucket/nm219/data/close_box_semimasked_4.pt",
    # "/vol/bitbucket/nm219/data/close_box_semimasked_5.pt",
    # "/vol/bitbucket/nm219/data/close_box_semimasked_6.pt",
    # "/vol/bitbucket/nm219/data/close_box_semimasked_7.pt",
    # "/vol/bitbucket/nm219/data/close_box_semimasked_8.pt",
    # "/vol/bitbucket/nm219/data/close_box_semimasked_9.pt",
    # "/vol/bitbucket/nm219/data/close_box_semimasked_10.pt"
]
output_file = '/vol/bitbucket/nm219/data/close_box_semimasked.pt'

merge_pt_files(input_files, output_file)