# how to the net

N : batch size

S : sequence size

the basic data structure:
```
seq = [img1, img2, ...]
seqs = [seq1, seq2, ...]
person = [seqs1, seqs2, ...]
datasets = [(person, label)]
id_to_folder = {'0': img_folder, '1':img_folder}
```
 

**how to get one sample**

1. pick 1 person randomly (every person has a id)
2. using the id to get the image list of that person
3. pick one picture from the image list randomly
4. 