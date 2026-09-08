def bump(mut x: Int):
    x += 1
def main():
    var data:List[Int]=[1, 2, 3]
    #ref first = data[0]
    #data.append(4)      # (a) why does using `first` after this fail?


    data.append(4)
    ref first = data[0]

    print(first)
    var n=len(data)
    # bump(len(data))     # (b) why won't this line compile?
    bump(n)
    print(n)