# class loop() : 

#     def __init__(self) : 

#         self.w = 8 
#         self.b = 2 

#     def __call__(self) : 

#         return self.w * self.b 


# l = loop() 

# print(l()) 
# print(vars(l).keys()) 
# print(vars(l).values())    

# print(dir(loop)) 

class ren() : 
    def __init__(self, a : int, b : float ) : 
        self.a = a 
        self.b = b 
        print(self.a, self.b) 

    


r = ren(a = 44.3, b =  45) 



