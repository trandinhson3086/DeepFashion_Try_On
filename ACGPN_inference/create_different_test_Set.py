import random

K = 16

with open("test_files/vton_test.txt", "r") as text_file:
    lines = text_file.readlines()

clothes = []
people = []
for l in lines:
    arr = l.strip().split(" ")
    clothes.append(arr[1])
    people.append(arr[0])

clothes = list(set(clothes))
people = list(set(people))

new_lines = []
for p in people:
    cs = random.choices(clothes, k=K)
    for c in cs:
        new_lines.append(p + " " + c + "\n")

with open("test_files/one_person_different_cloth.txt", "w") as text_file:
    lines = text_file.writelines(new_lines)
