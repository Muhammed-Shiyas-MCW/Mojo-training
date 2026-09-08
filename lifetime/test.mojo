struct Object(Copyable,Movable):
    var n:Int

    def __init__(out self, n:Int):
        self.n = n
        print("Init ",self.n)
    
    def __init__(out self, *, copy:Self):
        self.n= copy.n
        print("Copy ",self.n)
    
    def __init__(out self, *, deinit move:Self):
        self.n= move.n
        print("Move ",self.n)

    def __deinit__(deinit self):
        print("Deinit ",self.n)



def function(var x:Object) -> Object:
    return x^

def function2(n:Int)->Object:
    return Object(n)

def main():
    print("-init-")
    var x= Object(7)

    print("-use with function-")
    var y=function(x^)

    print("-explicit copy-")
    var p=y.copy()

    print("-transfer with ^-")
    var q=p^

    print(q.n)


    var r=function2(8)

    print(r.n)

    


