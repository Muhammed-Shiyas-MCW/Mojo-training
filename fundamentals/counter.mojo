from std.collections import Set

@fieldwise_init
struct EmptyError(Copyable,Writable):
    var message:String
    def write_to(self,mut writer:Some[Writer]):
        writer.write(self.message)

def word_count(text:String) raises-> Dict[String,Int]:
    var words=text.split()
    if len(words)==0:
        raise EmptyError("There are no words")
    var counter=Dict[String,Int]()
    var unique=Set[String]()
    for x in words:
        var word=String(x)
        counter[word]=counter.get(word,0)+1
        unique.add(word)

    print(unique)
    print(len(unique)," unique words")
    return counter^

def main():
    try:
        print(word_count("a v b a c"))
        print(word_count(""))
        print(word_count("     "))
    except e:
        print(e)

