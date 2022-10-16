# pytorch_leanring_framwork_from_scratch

### in this repo we will go over the basics of tensors manuipliation

* tensors intialization and Operations ; 

    * intialization :
        here there's many to intialize the tensor from troch but to keep in mind 
        tensors are just other format of intialize the matrcies in mulit-dimessions 
        and gives more flexibility to implement different analysis and calculs approchs 

        ```python

            import troch 

            x = torch.tensor([1,5,6]) ## this tensor is 1D
            print(x.ndimesions())
            print(x.shapoe)
            #########
            # troch.tensor() has some methods associated with it based on class  such :
            .shape() # gives the dimessions features of tensor
            .size() # size of tensor
            #########
            print(type(x))
            n=2 ; m=3 ; p=3 ## demissions 
            ## here this method is create random data point in 2D 
            y=troch.rand(n,m,size(2,3))
            ## this instaince empty Object tensor 
            empty_tensor=torch.empty(size(n,m,p),out=outs)
        ```

        here some differents ways to intialize tensor here almost tensors depends on 
        the ways uses cases , just to keep in mid that intialization 3D must be declare by knowing rows and cols compability to nor boradcasting dimessions 

        ```python

            my_tonsor_i = torch.empty(size=(3,3))
            print("tensor empyt",my_tonsor_i)
            my_tonsor_i = torch.zeros((3,3))
            print("tensor zeros",my_tonsor_i)
            my_tonsor_i = torch.ones((3,3))
            print("tensor ones",my_tonsor_i)
            my_tonsor_i = torch.rand((3,3))
            print("tensor rand",my_tonsor_i)
            my_tonsor_i = torch.eye(3,3)
            print("tensor eye",my_tonsor_i)
            my_tonsor_i = torch.linspace(start=3, end=2,steps=10)
            print("tensor linespace",my_tonsor_i)
            my_tonsor_i = torch.empty(3,3).uniform_(0,1)
            print("tensor mean/std",my_tonsor_i)

        ```
                    
                    '