# Decentralized Federated Learning
This is a Federated Learning implementation using the algorithm Federated Average. Video explanation can be found in this [video](https://youtu.be/umYhUZNY5Rc) (video is in portuguese).

Implemeted by: Beatriz Maia, Iago Cerqueira & Sophie Dilhon

## Running the application
### Environment Config
The application was made using Python 3.10 and there are a few libraries that you may need to install.
It is recommended to use a virtual environment, for that you may run the following commands:

```sh
python -m venv {environment}
```

where {environment} can be any name of your choice. After creating it, it has to be activated. On Linux and Mac, use the following command:

```sh
source /{environment}/bin/activate
```

and on Windows:

```sh
.\{environment}\Scripts\activate
```

Finally, install the dependencies with

```sh
pip install -r requirements.txt
```

### Execution
To execute the system, first run one of the following scripts. They are responsible to create the files used by the clients and server for the communication via grpc.

```sh
# Mac or Linux
./config.sh

# Windows
.\config.bat
```


To run the server and clients, execute the following commands in different terminals. 
```sh
python server.py --min_clients_per_round {n} --max_clients_total {m} --max_rounds {r} --accuracy_threshold {a} --timeout {t} --save_model --save_test
python client.py --ipv4 {i} --batch_size {b} --save_train --save_test
```

Server flags meaning:
--min_clients_per_round: Minimum number of clients per round.
--max_clients_total: Maximum number of clients per round.
--max_rounds: Maximum number of rounds.
--accuracy_threshold: Minimum accuracy threshold.
--timeout: Timeout in seconds for the server between training sessions.
--save_model: Save the model after training. This means that the server will use this model for next training sessions.
--save_test: Saves the test results in a csv file.

Client flags meaning:
--ipv4: IPv4 address of the client.
--save_train: Save the training results to csv file.
--save_test: Save the testing results to csv file.
--batch_size: Batch size for training.

Several clients can be created. The flags are not obligatory, the server will use default values if no argument is passed.

## Implementation

### Communication
To stablish the communication between server and clients and vice versa, the grpc lib was used, and two proto files were created. The first, [server.proto](proto/server.proto), is responsible for implementing the server's avaiable methods, and the other one, [client.proto](proto/client.proto), is responsible for implementing the clients's avaiable methods. For this implementation, the side that is resposible for calculating the centralized federated average is the one called the server, and the side responsible for training models "locally" are the clients.

#### Methods

The server only has one method, which is responsible for receiving and saving the client's data, and checking if the training can start.
```sh
add_trainer(trainer_request) returns (success);
```

As for the client, three methods were created. 
- The first is responsible for training the model with the local data, and it returns to the server the model's weights. 
    ```sh
    train_model(models_weights_input) returns (models_weights_output);
    ```
- The second one, receives the aggregated weights and tests the model with the local data.
    ```sh
    test_model(models_weights) returns (metrics_results);
    ```
- The last one ends the client's server
    ```sh
    finish_training(finish_message) returns (finish_message);
    ```

### Server
The server runs on `localhost:8080` and it is responsible for starting and finishing training. The code that represents it can be found at [server.py](server.py). The genneral logic behind it can be described by the following steps:

1. Server is initialized and connected to localhost:8080. It waits for clients to connect.
2. When `max_clients_total` connected to the server, it will start the training, sending to each of them the model weights, current round and number of clients training, as well as the training session id. To do that, the server must connect to the client's servers using their ipv4 and port (adress).
3. After training, each client sends back to the server their models weights.
4. The server aggregates the weights calculating the federated average, where models that trained with more samples are given more importance. This can be summarized by:
    ```py
    sum(weights * local_sample_size) / sum(local_sample_size)
    ```
5. The new weights are sent to every client (even non trainers), who then test the model and return the accuracy trained on their local data. 
6. Finally, the accuracies's mean is compared to the threshold, if it's smaller and current round < `max_rounds` then a new round starts. Else, training ends,  clients close their server and the server goes back to step 2.

In order for the server to be able to both receive request ad well as idely wait for training and start training, the usage of thread was necessary. When the `TrainerAgregator` class is initialized in the server, the following code is executed:

```py
threading.Thread(target=self.__train).start()
```

This makes it so that a second thread is running, executing the __train method. This method is the one responible for listening in and triggering a training session. In order to have a time difference between each training session, we added a timeout after a session is completed. This means that if a session was to be finilized and new clients were available, training would only begin after the timeout was finished.

An important decision made during the implementation was permiting new clients to join when a session has started. This means that while that client will not be participating in the current training round, it can be chosen for the next rounds.

The clients runs on a random generate port on `localhost`, however that can be changed if the ipv4 argument is configured elsewise. The code that represents each client can be found at [client.py](client.py). They end up having a type of behavior that is both of a client and a server. As a client, they add trainers to the server. As a server, they train the models and return the weights.

To simulate data on the client side, we use tensorflow MNIST and reduce the train and test datasets. During the train portion of the code, this can be observed by
```py
percentage = int(1 / (request.number_of_trainers + 10) * 100)
min_lim = min(5, percentage)
random_number = randint(min_lim, percentage) / 100

sample_size_train = int(random_number * len(self.x_train))
```
This section of code attempts to get a portion of the train data depending on how many trainers are currently in that round. In order for the clients to have an even smaller dataset, making it easier to see the models improvements between rounds in the analysis, the denominator is increased by 10.  In order to also have a variety of weight, where each client mught contribute differently to the federated average calculation, we added an extra randomizing step. This step can be translated into: choose a sample size that is between 5% and the percentage calculate beforehand. We added a min_lim just in case the percentage calculated before is smaller than 5%. 

For test, we consider the same size of data between all trainers. 
```py
sample_size_test = int((1/request.number_of_trainers)*len(self.x_test))
```

## Analysis

This analysis is done considering that 10 max trainers were used. For each round, the minimum amoun of trainers were 5, but more could be used. A total of 10 rounds were made.

In the image below, we can follow the training of the 10 different clients. The models start with a few data examples, which results in an accuracy of around 80%. Not all clients trained in the first round (or all rounds), only clients with an 'o' marker in round 0. 
![img](analysis/train_acc_000d635f-2206-4ab3-99b2-bd49a3c75fad.png)

After the federated average is calculated and sent to the new clients in round 1, the accuracy shots up. This shows that the usage of the algorithm helped tremendously, with an increase of 10%. This indicates that even though the clients between themselves had limited data, when calculting the average between them, it was possible to create a more accurate model. 

As the rounds increases, the accuracy as well, but in a slower ramp.


Analyzing the average test accuracy by the server size, we can also see this increase to the accuracy. While the training models start in rather lower percentage, the test considers the federate average calculated model, and it shows what observed in round 1 of training. Round 0 test results are already over 90%. As the rounds increases, so does the accuracy. 
![img](analysis/server_test_acc_000d635f-2206-4ab3-99b2-bd49a3c75fad.png)

Running for different of rounds, it's to observe how the accuracy increases, however it's not as big of an increase. The MNIST data is very simple, therefore this is expected.

![img](analysis/server_test_acc_10_20_40.png)

These results can be compared to [Lab 2](https://github.com/AHalic/SisDist_Labs/tree/main/Lab_2) results. While the traditional way of training, with all of the MNIST data, resulted into a near 100% accuracy, the federated average result was also extremely high. Our implementation had a similar result to the flwr implementation. 

![img](https://raw.githubusercontent.com/AHalic/SisDist_Labs/main/Lab_2/results_atv1/accuracy.png)