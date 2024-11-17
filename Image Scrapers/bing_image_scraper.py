from bing_image_downloader import downloader

# query contains search term
query = "Korean People Eating Korean Food"

# limit contains number of images to download
limit = 50

# output_dir contains path to save images
output_dir = "/media/jun/D47B-F7B9/People Eating Dataset"

# adult_filter_off: Set to false to enable adult content filter, true to disable adult content filter
adult_filter_off = False

# timeout contains time of the connection
timeout = 60

# run program
downloader.download(query, limit, output_dir, adult_filter_off, timeout = 120)

