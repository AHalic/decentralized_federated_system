import paho.mqtt.client as mqtt
import threading
import argparse
import hashlib
import random
import time
import uuid
import json

TIMEOUT_LIMIT = 60
QTDE_CLIENTS = 3

def parse_args() -> tuple[int, int]:
    """
    Parse command line arguments.

    Returns:
        tuple[int, int]: port, host
    """
    parser = argparse.ArgumentParser(description='Mine blocks serverless')
    parser.add_argument('--port', type=int, help='Port to listen to', default=1883)
    parser.add_argument('--host', type=str, help='Host to listen to', default='localhost')
    args = parser.parse_args()    

    return args.port, args.host

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

    def __init__(self, port, host):
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

        # Topicos:
        # sd/init: ClientID <int>
        # sd/voting: ClientID <int>, Vote <int>
        # sd/result: ClientID <int>, TransactionID <int>, Solution <string>, Result <int>
        # sd/challenge: TransactionID <int>, Challenge <string>
        client.subscribe([("sd/init", 0), ("sd/voting", 0), ("sd/result", 0), ("sd/challenge", 0)])
        threading.Thread(target=self.send_init).start()

    def send_init(self):
        """
        Send init message until all clients are known or timeout.
        """
        start = time.time()
        msg = json.dumps({"ClientID": self.uuid})

        while time.time() - start < TIMEOUT_LIMIT or len(self.known_clients) < QTDE_CLIENTS:
            self.client.publish("sd/init", msg)
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
        if msg.topic == "sd/init":
            self.on_init(msg.payload)

        elif msg.topic == "sd/voting":
            self.on_voting(msg.payload)

        elif msg.topic == "sd/result":
            self.on_result(msg.payload)

        elif msg.topic == "sd/challenge":
            self.on_challenge(msg.payload)

        elif msg.topic == "sd/solution":
            self.on_solution(msg.payload)

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
        client_id = json.loads(msg)['ClientID']
        print('mensagem de init', msg)

        if client_id not in self.known_clients and len(self.known_clients) < QTDE_CLIENTS:
            self.known_clients.append(client_id)
        if len(self.known_clients) == QTDE_CLIENTS:
            self.client.unsubscribe("sd/init")
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
            self.client.unsubscribe("sd/voting")
            print('Votação encerrada')
            
            self.elect_leader()

    def on_result(self, msg):
        """
        Function for when a result message is received from the MQTT broker.

        Args:
            msg (mqtt.MQTTMessage): Message.
        """
        msg_json = json.loads(msg)

        if msg_json['Result'] == 1 and msg_json['TransactionID'] == self.challenge['TransactionID']:
            print('\nSolução encontrada: ', msg_json['Solution'])
            print('Ganhador: ', msg_json['ClientID'])
            self.solution_found = True

    def on_challenge(self, msg):
        """
        Function for when a challenge message is received from the MQTT broker.
        """
        msg_json = json.loads(msg)
        self.challenge = msg_json
        self.solution_found = False

        print('\nNovo desafio recebido, dificuldade: ', msg_json['Challenge'])
        self.search_solution()

    def on_solution(self, msg):
        """
        Function for when a solution message is received from the MQTT broker.
        The controller validates the solution and sends a message to the broker saying if it is valid or not.

        Args:
            msg (mqtt.MQTTMessage): Message.
        """
        msg_json = json.loads(msg)

        challenge = self.challenges[msg_json['TransactionID']]

        result = {
            'ClientID': msg_json['ClientID'],
            'TransactionID': msg_json['TransactionID'],
            'Solution': msg_json['Solution'],
        }

        if self.validate_solution(msg_json['Solution'], challenge['Challenge']) and challenge['Winner'] == 0:
            result['Result'] = 1
            challenge['Winner'] = msg_json['ClientID']
            challenge['Solution'] = msg_json['Solution']

            print('\nSolução encontrada: ', msg_json['Solution'])
            print('Ganhador: ', msg_json['ClientID'])
        else:
            result['Result'] = 0

        self.client.publish("sd/result", json.dumps(result))

        if result['Result'] == 1:
            threading.Thread(target=self.show_menu).start()

    def validate_solution(self, solution, challenge):
        """
        Function to validate the solution of the challenge.

        Args:
            solution (int): Solution of the challenge.
            challenge (int): Challenge difficulty.

        Returns:
            bool: True if the solution is valid, False otherwise.
        """
        challenge_search = '0' * challenge
        result = bin(int(hashlib.sha1(str(solution).encode()).hexdigest(), 16))[2:].zfill(160)

        if result[:challenge] == challenge_search:
            return True
        else:
            return False

    def search_solution_thread(self, transaction_id, challenge):
        """
        Function to search for a solution to the challenge.
        
        Args:
            transaction_id (int): Transaction ID.
            challenge (int): Challenge difficulty.
        """
        attempts = set()
        while True:
            count = 0
            while count < 5:
                random_num = random.randint(0, 10000000)

                if random_num not in attempts:
                    attempts.add(random_num)
                    valid = self.validate_solution(random_num, challenge)
                    count += 1
                    
                    if valid:
                        
                        result = {
                            'TransactionID': transaction_id,
                            'ClientID': self.uuid,
                            'Solution': str(random_num)
                        }

                        print('Solução local encontrada: ', random_num)
                        if not self.solution_found:
                            self.client.publish("sd/solution", json.dumps(result))

                        return
                
            if self.solution_found:
                print('Ganharam a transação, interrompendo mineração')
                break

    def search_solution(self):
        """
        Function that creates Threads to search for a solution to the challenge.
        """
        threads = []
        for _ in range(6): 
            threads.append(threading.Thread(target=self.search_solution_thread, args=(self.challenge['TransactionID'], self.challenge['Challenge'])))

        # Start searching solutions
        for t in threads:
            t.start()


    def send_voting(self):
        """
        Function to send a voting message to the MQTT broker.
        """
        vote = uuid.uuid4()
        vote_msg = json.dumps({"ClientID": self.uuid, "VoteID": str(vote)})

        self.client.publish("sd/voting", vote_msg)    

    def elect_leader(self):
        """
        Function to elect the controller.
        If the client is the controller, it unsubscribes from the voting topic, subscribes to the challenge topic
        and shows the menu.
        """

        # finds vote of biggest value, if there is a tie, client ID is used as tie breaker
        max_vote = max(self.votes, key=lambda x: (x['VoteID'], x['ClientID']))

        if max_vote['ClientID'] == self.uuid:
            self.client.unsubscribe(["sd/result", "sd/challenge"])
            
            self.client.subscribe("sd/solution")
            self.challenges = []

            self.show_menu()
         
    def show_menu(self):
        """
        Function to show the controller's menu.
        """
        print("\nEscolha uma das opções abaixo:")
        print("1 - Iniciar desafio")
        print("2 - Encerrar")

        option = int(input("Opção: "))
        if option == 1:
            self.new_transaction()
        elif option == 2:
            self.client.disconnect()

    def new_transaction(self):
        """
        Function to create a new transaction.
        """
        difficulty = random.randint(10, 20)
        print(f"New challenge: lvl {difficulty}")
        
        challenge = {
            "TransactionID": len(self.challenges),
            "Challenge": difficulty,
            "Solution": None,
            "Winner": 0
        }
        self.challenges.append(challenge)

        self.client.publish("sd/challenge", json.dumps({"TransactionID": challenge["TransactionID"], "Challenge": challenge["Challenge"]}))


if __name__ == "__main__":
    port, host = parse_args()
    client = Client(port, host)