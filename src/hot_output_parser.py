from math import floor

class HotParser:
    def __init__(self):
        self.backslashBuffer = 0
        self.quoteCount = 0
        self.nextChrIsGuaranteedToNotBeTheEnd = False
        self.inTheAnswer = False

    def __HandleNormalCharacter(self, chr: str, printableChunk: str) -> str:
        backslashesBuffered = ("\\" * self.backslashBuffer)
        printableChunk += backslashesBuffered+chr
        self.backslashBuffer = 0
        self.nextChrIsGuaranteedToNotBeTheEnd = False
        return printableChunk

    def __HandleBackslash(self):
        self.backslashBuffer += 1
        if self.backslashBuffer%2 == 0: self.nextChrIsGuaranteedToNotBeTheEnd = False
        else: self.nextChrIsGuaranteedToNotBeTheEnd = True

    def __HandleQuote(self, printableChunk: str, done: bool) -> tuple[str, bool]:
        totalBackslashesToPrint = floor(self.backslashBuffer/2)
        printableChunk += ("\\" * totalBackslashesToPrint)
        self.backslashBuffer = 0
        if self.nextChrIsGuaranteedToNotBeTheEnd:
            printableChunk += "\""
            self.nextChrIsGuaranteedToNotBeTheEnd = False
        else:
            self.quoteCount = 0
            self.inTheAnswer = False
            done = True
        return printableChunk, done
    
    def HotParseChunk(self, chunk: str) -> tuple[str, bool]:
        printableChunk = ""
        done = False
        if type(chunk) != str:
            return printableChunk, done
        elif len(chunk) == 0:
            return printableChunk, done
        for chr in chunk:
            if not self.inTheAnswer:
                if chr == "\"":
                    self.quoteCount += 1
                    if self.quoteCount == 3: self.inTheAnswer = True
            else:
                if chr != "\\" and chr != "\"":
                    printableChunk = self.__HandleNormalCharacter(chr, printableChunk)
                elif chr == "\\":
                    self.__HandleBackslash()
                else:
                    printableChunk, done = self.__HandleQuote(printableChunk, done)
                    if done: break
                    
        return printableChunk, done