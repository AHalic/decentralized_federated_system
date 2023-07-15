# Decentralized Federated Learning
This is a Decentralized Federated Learning implementation using the algorithm Federated Average. Video explanation can be found in this [video]() (video is in portuguese).

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

For communication between clients a broker is created, make sure you have Docker installed for that and run it with
```sh
docker compose up
```
The broker will be avaible on port `1883`.

### Execution

To run the clients, execute the following commands in different terminals (at least `{qtd_clients}`). 
```sh
python client.py --host {h} --port {p} --batch_size {b} --qtd_clients {k} --train_clients {n} --accuracy_threshold {a} --max_rounds {r}
```

Client flags meaning:
--port: Port on which the broker is avaible
--host: Host on which the broker is avaible
--batch_size: Batch size for training.
--qtd_clients: Number of clients connected to the broker.
--train_clients: Number of clients training per round.
--max_rounds: Maximum number of rounds.
--accuracy_threshold: Minimum accuracy threshold.


Several clients can be created, it must be at least `qtd_clients`. The flags are not obligatory, the server will use default values if no argument is passed.

## Implementation

### Communication
The communication between clients is done using the Publisher/Subscriber model (emqx mqtt broker). To stablish a connection all clients must send a `InitMsg` with their ID's. This makes them become a known client.


### Election
For this implementation, the side that is resposible for calculating the centralized federated average will be called as server, and the side responsible for training models "locally" as clients, though initially every process is a client. One client must then be elected as server, for that every known client sends a `ElectionMsg` with their vote, the biggest becomes the server.

### How it works
The server is responsible for starting and finishing training. After being elected it will start the training, sending a `TrainingMsg` containing the model weights, current round and the clients that will train ID's, as well as the training session id.

After training, each client sends a `RoundMsg` with their models weights, sample size and session id. The server reads every message and aggregates the weights calculating the federated average, where models that trained with more samples are given more importance. This can be summarized by:

    ```py
    sum(weights * local_sample_size) / sum(local_sample_size)
    ```
The server publishes an `AggregationMsg` message with the new weights, all clients (even non trainers) test the model and send an `EvaluationMsg` message with the accuracy obtained on their local data.

Finally the server analyse if the mean accuracy is equal or greater than the threshold or if the max round is reached. If none of that is applicable, a new round begins. If not, a `FinishMsg` is sent containing the word 'stop' and the training is finished. Later on a new election begins to restart the training process.

Data is divided as in [previous implementation of federated learning](https://github.com/beamaia/federated_system_mnist).

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