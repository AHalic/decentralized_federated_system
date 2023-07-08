import paho.mqtt.client as mqtt
import threading
import datetime
import argparse
import hashlib
import random
import time
import uuid
import json
import os

import tensorflow as tf
from tensorflow.keras.optimizers.legacy import SGD
import numpy as np

from model import ModelBase64Encoder, ModelBase64Decoder, define_model

TIMEOUT_LIMIT = 60
QTDE_CLIENTS = 3

def parse_args() -> tuple[int, int]:
    """
    Parse command line arguments.

    Returns:
        tuple[int, int]: port, host
    """
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('--port', type=int, help='Port to listen to', default=1883)
    parser.add_argument('--host', type=str, help='Host to listen to', default='broker.emqx.io')
    parser.add_argument('--save_train', action='store_true', default=False, help='Plot the training results')
    parser.add_argument('--save_test', action='store_true', default=False, help='Plot the testing results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--train_clients', type=int, default=2, help='Amount of clients to train')
    parser.add_argument('--min_clients_per_round', type=int, default=3, help='Minimum number of clients per round.')
    parser.add_argument('--max_rounds', type=int, default=10, help='Maximum number of rounds.')
    parser.add_argument('--accuracy_threshold', type=float, default=0.99, help='Minimum accuracy threshold.')
    args = parser.parse_args()    

    return args

class Client:
    """
    Client class.

    Attributes:
        uuid (str): Client's UUID.
        port (int): Port to connect to.
        host (str): Host to connect to.
        known_clients (list[str]): List of known clients.
        client (mqtt.Client): MQTT client.
    """
    uuid = str(uuid.uuid4())
    votes = []
    weights = []
    metrics = []

    def __init__(self, port, host, store_training_data = False, store_test_data = False, \
                 batch_size = 32, train_clients = 3, \
                 min_clients_per_round=5, max_rounds=10, \
                 accuracy_threshold = 0.90, timeout=300):
        """
        Constructor.

        Args:
            port (int): Port to connect to.
            host (str): Host to connect to.
        """
        print('UUID: ', self.uuid)
        self.port = port
        self.host = host
        self.known_clients = [self.uuid]

        self.min_clients_per_round = min_clients_per_round
        self.max_rounds = max_rounds
        self.accuracy_threshold = accuracy_threshold
        self.n = train_clients
        self.store_training_data = store_training_data
        self.store_test_data = store_test_data
        self.timeout = timeout

        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

        self.model = define_model((28, 28, 1), 10)
        self.opt = SGD(learning_rate=0.01, momentum=0.9)
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.batch_size = batch_size

        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.client.connect(host, port, 60)

        self.client.loop_forever()


    def on_connect(self, client, userdata, flags, rc):
        """
        Callback for when a connection is established with the MQTT broker.

        Args:
            client (mqtt.Client): MQTT client.
            userdata (Any): User data.
            flags (dict): Flags.
        """
        print("Connected with result code "+str(rc))

        client.subscribe([("InitMsg", 0), ("ElectionMsg", 0), ("TrainingMsg", 0), ("AggregationMsg", 0), ("FinishMsg", 0)])
        threading.Thread(target=self.send_init).start()

    def send_init(self):
        """
        Send init message until all clients are known or timeout.
        """
        start = time.time()
        msg = json.dumps({"ClientID": self.uuid})

        while time.time() - start < TIMEOUT_LIMIT or len(self.known_clients) < QTDE_CLIENTS:
            self.client.publish("InitMsg", msg)
            time.sleep(10)


    def on_message(self, client, userdata, msg):
        """
        Callback for when a message is received from the MQTT broker.
        Depending on the topic of the message calls the appropriate function.

        Args:
            client (mqtt.Client): MQTT client.
            userdata (Any): User data.
            msg (mqtt.MQTTMessage): Message.
        """

        # Topicos:
        # InitMsg: ClientID <int>
        # ElectionMsg: ClientID <int>, Vote <int>
        # TrainingMsg: [ClientID <int>] -> agregador
        # RoundMsg: pesos -> clientes
        # AggregationMsg: pesos agregados -> agregador
        # EvaluationMsg: resultados -> clientes
        # FinishMsg: finalizado -> agregador

        if msg.topic == "InitMsg":
            self.on_init(msg.payload)

        elif msg.topic == "ElectionMsg":
            self.on_voting(msg.payload)

        elif msg.topic == "TrainingMsg":
            self.on_training(msg.payload)

        elif msg.topic == "RoundMsg":
            self.on_round(msg.payload)

        elif msg.topic == "AggregationMsg":
            self.on_aggregation(msg.payload)

        elif msg.topic == "EvaluationMsg":
            self.on_evaluation(msg.payload)

        elif msg.topic == "FinishMsg":
            self.on_finish(msg.payload)

    def on_init(self, msg):
        """
        Function for when a init message is received from the MQTT broker.
        If the client is not known and the number of known clients is less than the number of clients needed,
        the client is added to the list of known clients.
        If the number of known clients is equal to the number of clients needed, the client unsubscribes from the init topic
        and starts the voting.

        Args:
            msg (mqtt.MQTTMessage): Message.
        """
        print('mensagem de init', msg)
        client_id = json.loads(msg)['ClientID']

        if client_id not in self.known_clients and len(self.known_clients) < QTDE_CLIENTS:
            self.known_clients.append(client_id)
        if len(self.known_clients) == QTDE_CLIENTS:
            self.client.unsubscribe("InitMsg")
            print('iniciando votação')
            self.send_voting()

    def on_voting(self, msg):
        """
        Function for when a voting message is received from the MQTT broker.
        If the number of known clients is equal to the number of clients needed, the client unsubscribes from the voting topic
        and elect the controller.

        Args:
            msg (mqtt.MQTTMessage): Message.
        """
        msg_json = json.loads(msg)
        print('mensagem de voto', msg)
        
        if msg_json['ClientID'] not in self.votes:
            self.votes.append(msg_json)

        if len(self.votes) == QTDE_CLIENTS:
            self.client.unsubscribe("ElectionMsg")
            print('Votação encerrada')
            
            self.elect_leader()

    def on_training(self, msg):
        # if self.uuuid in message, trains model
        print('mensagem de treinamento', msg)

        # clients = json.loads(msg)['clients']
        # weights = [np.array(weight) for weight in json.loads(msg)['weights']]
        # round = json.loads(msg)['round']
        # if self.uuid in clients:
        #     self.train_model(weights, round)


    def on_round(self, msg):
        self.weights.append({'weights': json.loads(msg)['weights'], 'sample_amount': json.loads(msg)['sample_amount']})
        
    def on_aggregation(self, msg):
        pass

    def on_evaluation(self, msg):
        pass

    def on_finish(self, msg):
        if msg == 'stop':
            self.__restart_config()
            threading.Thread(target=self.send_init).start()

    def send_voting(self):
        """
        Function to send a voting message to the MQTT broker.
        """
        vote = uuid.uuid4()
        vote_msg = json.dumps({"ClientID": self.uuid, "VoteID": str(vote)})

        self.client.publish("ElectionMsg", vote_msg)    

    def elect_leader(self):
        """
        Function to elect the controller.
        If the client is the controller, it unsubscribes from the voting topic, subscribes to the challenge topic
        and shows the menu.
        """

        # finds vote of biggest value, if there is a tie, client ID is used as tie breaker
        max_vote = max(self.votes, key=lambda x: (x['VoteID'], x['ClientID']))

        if max_vote['ClientID'] == self.uuid:
            self.client.unsubscribe(["TrainingMsg", "AggregationMsg", "FinishMsg"])
            
            self.client.subscribe([("RoundMsg", 0), ("EvaluationMsg", 0)])
            self.challenges = []

            # send message to trainingMsg topic
            print('Lider eleito, iniciando treinamento')
            self.start_training()

    # AGGREGATOR FUNCTIONS
         
    def start_training(self):
        # chooses n random clients to train except the controller
        possible_clients = self.known_clients.copy()
        possible_clients.remove(self.uuid)

        self.uuid = str(uuid.uuid4())
        print(f"[{datetime.datetime.now()}] Generating uuid for training session.")
        print(f"[{datetime.datetime.now()}] UUID: {self.uuid}")

        for round in range(self.max_rounds):
            print(f"[{datetime.datetime.now()}]********************************************************")
            print(f"[{datetime.datetime.now()}] Starting round {round+1}/{self.max_rounds}.")
            
            self.current_round = round
            clients = random.sample(possible_clients, self.n)

            # serialized_weights = [weight.tolist() for weight in self.model.get_weights()]

            # sends training message to chosen clients
            # self.client.publish("TESTE", json.dumps({"clients": str(clients), "round": str(self.current_round), "weights": str(serialized_weights)}))

            while len(self.weights) < self.n:
                pass

            avg_weights = self.__federeated_train()
            self.model.set_weights(avg_weights)

            self.client.publish("AggregationMsg", json.dumps({"weights": avg_weights }))
        
            while len(self.metrics) < self.known_clients:
                pass

            accuracy_mean = np.mean(self.metrics)
            print(f"[{datetime.datetime.now()}] Mean accuracy: {accuracy_mean * 100: 0.2f}.")

            # if self.save_test:
            #     self.__save_test(accuracy_mean)

            if accuracy_mean >= self.accuracy_threshold:
                print(f"[{datetime.datetime.now()}] Accuracy threshold reached. Stopping rounds.")
                break
        
        print(f"[{datetime.datetime.now()}] Training finished.")

        self.client.subscribe(["TrainingMsg", "AggregationMsg", "FinishMsg"])    
        self.client.unsubscribe(["RoundMsg", "EvaluationMsg"])

        self.__restart_config()
        self.__finish_training()

        threading.Thread(target=self.send_init).start()

    def load_data(self):
        """
        Loads the MNIST dataset

        Returns
        -------
        tuple
            Returns the training and testing data and labels
        """
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_train = x_train / 255.0
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        x_test = x_test / 255.0

        y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
        y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
    
        return (x_train, y_train), (x_test, y_test)

    def __finish_training(self):
        self.client.publish("FinishMsg", "stop")

    def __restart_config(self):
        """
        Restarts the server configuration.
        """
        self.known_clients = [self.uuid]
        self.votes = []
        self.weights = []
        self.metrics = []

        self.client.subscribe([("InitMsg", 0), ("ElectionMsg", 0)])

        if not self.save_model:
            self.model = define_model((28,28,1), 10)   
        self.current_round = 0 

    def __federeated_train(self):
        """
        Calculates the federated average of the weights.

        Returns
        -------
            list
                List of aggregated weights.
        """
        print(f"[{datetime.datetime.now()}] Calculating federated average.")

        weights_list = [result.weights for result in self.weights]
        sample_sizes = [result.sample_amount for result in self.weights]

        new_weights = []
        for layers in zip(*weights_list):
            aggreagation = []
            for layer, sample_size in zip(layers, sample_sizes):
                if isinstance(aggreagation, list) and not len(aggreagation):
                    aggreagation = layer * sample_size
                else:
                    aggreagation += layer * sample_size
            new_layer = aggreagation / sum(sample_sizes)
            new_weights.append(new_layer)
            
        return new_weights


    # TRAINERS FUNCTIONS
    def train_model(self, weights, round):
        """
        Trains the model

        Parameters
        ----------
        request : client_pb2.models_weights_input
            Request containing the model weights, current round and number of trainers

        Returns
        -------
        client_pb2.models_weights_output
            Returns the model weights and the sample size used for training
        """
        print(f"[{datetime.datetime.now()}]********************************************************")
        print(f"[{datetime.datetime.now()}] Training model. Round number: " + str(round))
 
        percentage = int(1 / (self.n + 10) * 100)
        min_lim = min(5, percentage)
        random_number = random.randint(min_lim, percentage) / 100
        
        sample_size_train = int(random_number * len(self.x_train))
        print(f"[{datetime.datetime.now()}] Sample size: {sample_size_train}")

        idx_train = np.random.choice(np.arange(len(self.x_train)), sample_size_train, replace=False)
        x_train = self.x_train[idx_train]
        y_train = self.y_train.numpy()[idx_train]

        model_weights = ModelBase64Decoder(weights)
        self.model.set_weights(model_weights)

        history = self.model.fit(x_train, y_train, batch_size=self.batch_size ,epochs=1, verbose=False)
        model_weights = ModelBase64Encoder(self.model.get_weights())

        print(f"[{datetime.datetime.now()}] Training finished. Results:")
        print(f"[{datetime.datetime.now()}] Accuracy: {history.history['accuracy'][0]}")
        print(f"[{datetime.datetime.now()}] Loss: {history.history['loss'][0]}")

        # if self.store_training_data:
        #     self.store_information(history.history['loss'][0], history.history['accuracy'][0], round)
        
        self.client.publish("RoundMsg", json.dumps({"weights": model_weights, "sample_amount": sample_size_train}))

    def test_model(self, weights):
        """
        Tests the model

        Parameters
        ----------
        request : client_pb2.models_weights_input
            Request containing the model weights, current round, number of trainers and uuid of the training session

        Returns
        -------
        client_pb2.metrics_results
            Returns the accuracy of the model
        """
        print(f"[{datetime.datetime.now()}]********************************************************")
        print(f"[{datetime.datetime.now()}] Testing model.")

        sample_size_test = int((1/self.n)*len(self.x_test))
        idx_test = np.random.choice(np.arange(len(self.x_test)), sample_size_test, replace=False)
        x_test = self.x_test[idx_test]
        y_test = self.y_test.numpy()[idx_test]
        
        model_weights = ModelBase64Decoder(weights)
        self.model.set_weights(model_weights)
        results = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=False)

        print(f"[{datetime.datetime.now()}] Testing finished. Results:")
        print(f"[{datetime.datetime.now()}] Accuracy: {results[1]}")
        print(f"[{datetime.datetime.now()}] Loss: {results[0]}")

        # if self.store_test_data:
        #     self.store_information(results[0], results[1], round, train=False)

        self.client.publish("EvaluationMsg", json.dumps({"accuracy": results[1]}))

    def store_information(self, loss, accuracy, round, session_uuid, train=True):
        """
        Stores the training and testing results in a csv file

        Parameters
        ----------
        loss : float
            Loss of the model
        accuracy : float
            Accuracy of the model
        round : int
            Round number
        session_uuid : str
            Unique identifier of the training session
        train : bool
            Flag to indicate if the results are from training or testing
        """

        now = datetime.datetime.now()
        
        file_name = "train" if train else "test"
        file_name += "_client_results.csv"

        new_file = os.path.isfile(file_name)

        headers = ['uuid', 'ipv4','port', 'accuracy', 'loss', 'round', 'timestamp', 'session_uuid']

        with open (file_name,'a') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
            if not new_file:
                writer.writeheader()
            writer.writerow({'uuid': self.uuid, 
                             'ipv4': self.ipv4, 
                             'port': self.port, 
                             'accuracy':accuracy, 
                             'loss': loss, 
                             'round': round, 
                             'timestamp': now, 
                             'session_uuid': session_uuid})



if __name__ == "__main__":
    args = parse_args()
    client = Client(port=args.port, host=args.host, store_training_data=args.save_train, store_test_data=args.save_test, \
                    batch_size=args.batch_size, train_clients=args.train_clients, \
                    min_clients_per_round=args.min_clients_per_round, max_rounds=args.max_rounds, \
                    accuracy_threshold=args.accuracy_threshold)