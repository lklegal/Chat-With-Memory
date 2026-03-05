from math import floor

class HotParser:
    def __init__(self):
        self.backslashBuffer = 0
        self.quoteCount = 0
        self.nextChrIsGuaranteedToNotBeTheEnd = False
        self.inTheAnswer = False
    
    def HotParseChunk(self, chunk: str) -> tuple[str, bool]:
        printableChunk = ""
        done = False
        for chr in chunk:
            if not self.inTheAnswer:
                if chr == "\"":
                    self.quoteCount += 1
                    if self.quoteCount == 3: self.inTheAnswer = True
            else:
                if chr != "\\" and chr != "\"":
                    backslashesBuffered = ""
                    for _ in range(self.backslashBuffer):
                        backslashesBuffered += "\\"
                    self.backslashBuffer = 0
                    printableChunk += backslashesBuffered+chr
                    self.nextChrIsGuaranteedToNotBeTheEnd = False
                elif chr == "\\":
                    self.backslashBuffer += 1
                    if self.backslashBuffer%2 == 0: self.nextChrIsGuaranteedToNotBeTheEnd = False
                    else: self.nextChrIsGuaranteedToNotBeTheEnd = True
                    continue
                else:
                    totalBackslashesToPrint = floor(self.backslashBuffer/2)
                    backslashesBuffered = ""
                    for _ in range(totalBackslashesToPrint):
                        backslashesBuffered += "\\"
                    printableChunk += backslashesBuffered
                    self.backslashBuffer = 0
                    if self.nextChrIsGuaranteedToNotBeTheEnd:
                        printableChunk += "\""
                        self.nextChrIsGuaranteedToNotBeTheEnd = False
                    else:
                        self.quoteCount = 0
                        self.inTheAnswer = False
                        done = True
                        break
        return printableChunk, done