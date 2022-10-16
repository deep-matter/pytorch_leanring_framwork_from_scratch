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
    * Operations  :     

        almost mathematical operation done top on tensor are 

        * mulitplication :
            multiplay matrix is often used in many process such Neural netwrok 

            ```python 
                import troch 

                x=torch.empty((2,2)).uniform_(0,1) ## uniform distrubtin 
                y=troch_empty((2,2)).normal_(1,4)   ### normal distrbtion 
                out=troch.mul(x,y)
                ## other way 
                ## but there's a little different in between them 
                # troch.mm is used for matrix shape of ( nxm) * (mxp) --> (nxp)
                # here in this the error will raise if you change the shapo D 
                out_=torch.mm(x,y)
                # exponotional matrix 
                ##########
                # # additional two 1D matrix
                tensor_x = torch.rand(2,2)
                tensor_y = torch.rand(2,2)
                dot_product=torch.dot(tensor_x,tensor_y)
                print(dot_product)
                power_matrix = torch.matrix_power(matrix_x, 2)

            ```
                       
        * Slicing a 3D Tensor

            Slicing: Slicing means selecting the elements present in the tensor by using “:” slice operator. We can slice the elements by using the index of that particular element.
            Note: Indexing starts with 0

            Syntax: tensor[tensor_position_start:tensor_position_end, tensor_dimension_start:tensor_dimension_end , tensor_value_start:tensor_value_end]

            * Parameters:

                tensor_position_start: Specifies the Tensor to start iterating
                tensor_position_end: Specifies the Tensor to stop iterating
                tensor_dimension_start: Specifies the Tensor to start the iteration of tensor in given positions
                tensor_dimension_stop: Specifies the Tensor to stop the iteration of tensor in given positions
                tensor_value_start: Specifies the start position of the  tensor to iterate the elements given in dimensions
                tensor_value_stop: Specifies the end position of the tensor to iterate the elements given in dimensions

                ```python
                batch_size =3
                features = 3
                x = torch.tensor([[1,2,3],[3,5,6],[5,9,9]],device='cuda')
                z=torch.rand(12,24,14)
                y=torch.randint(2, (5,5))
                print(z.ndimension())

                # print(x[0:3,1:2])
                # print(x[x.remainder(2)==0])
                #print(x)

                #print(x[0:, 0:13])

                # x1 = torch.rand(3,4)
                # row = torch.tensor([1,2])
                # col =torch.tensor([2,3])

                # print(x1[row,col].shape)

                ``` 
