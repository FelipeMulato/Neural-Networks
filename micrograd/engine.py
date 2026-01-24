import math
class Value:
    def __init__(self,data,_children=(),_op=''):
        self.data =data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out
    
    def __sub__ (self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data-other.data,(self,other),'-')
        
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += (-1.0)*out.grad
        out._backward = _backward
         
        return out
    def __mul__ (self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data,(self,other),'*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data/other.data,(self,other),'/')

        def _backward():
            self.grad += (1.0/other.data)* out.grad
            other.grad += (-1.0*(self.data/math.sqrt(other.data)))*out.grad
        out._backward = _backward
        return out
    
    def __pow__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data**other.data,(self,other),'**')

        def _backward():
            self.grad +=  (other.data)*((self.data)**(other-1))*out.grad
            
        out._backward = _backward
        return out

    
    def tanh(self):
        x= (math.exp(2*self.data)-1.0)/(math.exp(2*self.data)+1.0)
        out = Value(x,(self,),'tanh')

        def _backward():
            self.grad =  (1 - math.pow(x,2))*out.grad
        
        out._backward = _backward
        return out
    
    def backaward(self):
        topo = []
        visited = set()

        def topo_sort(node):
            visited.add(node)
            for next in node._prev:
                if next not in visited:
                    topo_sort(next)
            topo.append(node)
        topo_sort(self)
        self.grad =1.0
        for node in reversed(topo):
            node._backward()
