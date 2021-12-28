from hentai import Hentai, Format

doujin = Hentai(177013)

# True
Hentai.exists(doujin.id)

# METAMORPHOSIS
print(doujin.title(Format.Pretty))

# [Tag(id=3981, type='artist', name='shindol', url='https://nhentai.net/artist/shindol/', count=100000)]
print(doujin.artist)

# ['dark skin', 'group', ... ]
print([tag.name for tag in doujin.tag])

# 2016-10-18 12:28:49+00:00
print(doujin.upload_date)

# ['https://i.nhentai.net/galleries/987560/1.jpg', ... ]
print(doujin.image_urls)

# get the source
doujin.download(progressbar=True)