def logger[*ArgType:Writable](*msgs:*ArgType, sep:String=" ", level:String="info")->String:
    var log:String = "[" + level.upper() +"] "
    comptime n=msgs.__len__()
    comptime for i in range(n):
        log+=String(msgs[i])
        comptime if i < n-1:
            log+=sep
    return log

def main():
    print(logger("test",10,10.8))
    print(logger("loss",0.42,"step",10,level="DEBUG"))
    print(logger('loss',100,10.4,level="info",sep=" -> "))