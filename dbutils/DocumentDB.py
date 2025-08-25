from pymongo import MongoClient, ASCENDING, TEXT
from sshtunnel import SSHTunnelForwarder
import pprint
import threading
import os

class DocumentDB:
    _instances = {}
    _lock = threading.Lock()
    
    # DocumentDB endpoints for different targets
    DOCDB_ENDPOINTS = {
        'web2': 'aiseer-documentdb-1.c3gsycya27j9.us-west-2.docdb.amazonaws.com',
        'api': 'aiseer-documentdb-2.c3gsycya27j9.us-west-2.docdb.amazonaws.com',
        'web3': 'aiseer-documentdb-3.c3gsycya27j9.us-west-2.docdb.amazonaws.com'
    }
    
    def __new__(cls, name='web2'):
        if name not in cls._instances:
            with cls._lock:
                if name not in cls._instances:
                    cls._instances[name] = super(DocumentDB, cls).__new__(cls)
        return cls._instances[name]
    
    def __init__(self, name='web2'):
        if not hasattr(self, '_initialized'):
            # ========== Configuration ==========
            # Bastion EC2 details
            self.BASTION_HOST = '54.185.32.171'
            self.BASTION_USER = 'ubuntu'
            self.BASTION_KEY_PATH = 'aiseer-pub-prod-key.pem'  # Private key file in root directory

            # DocumentDB internal endpoint based on name
            self.name = name
            self.DOCDB_HOST = self.DOCDB_ENDPOINTS.get(name, self.DOCDB_ENDPOINTS['web2'])
            self.DOCDB_PORT = 27017

            # Auth
            self.DB_USERNAME = 'docdbadmin'
            self.DB_PASSWORD = 'ZXY0NcwP78Pfmqv'
            self.DB_NAME = 'admin'

            # Path to CA bundle
            self.CA_CERT_PATH = 'global-bundle.pem'  # CA certificate file in root directory
            
            # Connection state
            self._client = None
            self._tunnel = None
            
            self._initialized = True
    
    def connect(self):
        """Establish connection to DocumentDB through SSH tunnel"""
        if self._client is not None:
            print(f"🔄 Reusing existing connection to {self.name} DocumentDB")
            return self._client
            
        try:
            print(f"🔌 Connecting to {self.name} DocumentDB: {self.DOCDB_HOST}")
            print(f"📡 Using SSH tunnel through {self.BASTION_HOST}")
            
            # ========== Create SSH Tunnel and Connect ==========
            # Use different local ports for each target to avoid conflicts
            local_ports = {
                'web2': 27020,
                'api': 27021,
                'web3': 27022
            }
            local_port = local_ports.get(self.name, 27017)
            print(f"🔗 Local port: {local_port}")
            
            self._tunnel = SSHTunnelForwarder(
                (self.BASTION_HOST, 22),
                ssh_username=self.BASTION_USER,
                ssh_private_key=self.BASTION_KEY_PATH,
                remote_bind_address=(self.DOCDB_HOST, self.DOCDB_PORT),
                local_bind_address=('localhost', local_port)
            )
            
            print("🚇 Starting SSH tunnel...")
            self._tunnel.start()
            print(f"✅ SSH tunnel established on localhost:{self._tunnel.local_bind_port}")
            
            print("🔐 Creating MongoDB client with retryWrites=False...")
            self._client = MongoClient(
                host='localhost',
                port=self._tunnel.local_bind_port,
                username=self.DB_USERNAME,
                password=self.DB_PASSWORD,
                tls=True,
                tlsCAFile=self.CA_CERT_PATH,
                authSource='admin',
                tlsAllowInvalidHostnames=True,
                directConnection=True,
                retryWrites=False
            )

            # Test connection
            print(f"🧪 Testing connection to {self.name}...")
            server_info = self._client.server_info()
            print(f"✅ Connected successfully to {self.name}!")
            print(f"📊 Server version: {server_info.get('version', 'Unknown')}")
            print(f"🔧 Server features: {server_info.get('features', {})}")

            # List all databases
            print(f"\n📂 Databases available on {self.name}:")
            db_names = self._client.list_database_names()
            for db_name in db_names:
                print(f"  - {db_name}")
            print(f"📊 Total databases: {len(db_names)}")
            
            return self._client
            
        except Exception as e:
            print(f"❌ Error connecting to {self.name}:", e)
            if self._tunnel:
                self._tunnel.stop()
            raise

    def create_indexes(self, db):
        """Add an index to a collection"""
        print("🔍 Creating database indexes...")
        
        # 1. bonus_credit_tracker
        print("  📊 Creating indexes for bonus_credit_tracker...")
        db.bonus_credit_tracker.create_index([("task_id", ASCENDING)])
        db.bonus_credit_tracker.create_index([("timestamp", ASCENDING)])

        # 2. claim_context_map
        print("  📊 Creating indexes for claim_context_map...")
        db.claim_context_map.create_index([("claim_text", TEXT)])

        # 3. claims
        print("  📊 Creating indexes for claims...")
        db.claims.create_index([("url", ASCENDING)])

        # 4. mini_search
        print("  📊 Creating indexes for mini_search...")
        db.mini_search.create_index([
            ("title", TEXT),
            ("body", TEXT),
            ("tags", TEXT),
            ("link", TEXT)
        ], name="title_body_tags_link_text_index")


        # 5. query_new
        print("  📊 Creating indexes for query_new...")
        db.query_new.create_index([("dislikes", ASCENDING)])
        db.query_new.create_index([("likes", ASCENDING)])
        db.query_new.create_index([("query", ASCENDING)])
        db.query_new.create_index([("task_id", ASCENDING)])

        # 6. tweets
        print("  📊 Creating indexes for tweets...")
        db.tweets.create_index([("stage", TEXT)])

        # 7. url_content
        print("  📊 Creating indexes for url_content...")
        db.url_content.create_index([("link", ASCENDING)])
        
        print("✅ All indexes created successfully!")

    
    def get_client(self):
        """Get the MongoDB client instance"""
        if self._client is None:
            print(f"🔄 No existing client for {self.name}, creating new connection...")
            return self.connect()
        print(f"✅ Returning existing client for {self.name}")
        return self._client
    
    def get_database(self, db_name):
        """Get a specific database"""
        client = self.get_client()
        return client[db_name]
    
    def list_databases(self):
        """List all available databases"""
        client = self.get_client()
        print(f"\n📂 Databases available on {self.name}:")
        for db_name in client.list_database_names():
            print("-", db_name)
        return client.list_database_names()
    
    def close(self):
        """Close the database connection and SSH tunnel"""
        if self._client:
            print(f"🔌 Closing MongoDB client for {self.name}")
            self._client.close()
            self._client = None
        
        if self._tunnel:
            print(f"🚇 Stopping SSH tunnel for {self.name}")
            self._tunnel.stop()
            self._tunnel = None
        
        print(f"✅ Successfully closed all connections for {self.name}")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# Create global instances for each target
document_db_web2 = DocumentDB('web2')
document_db_api = DocumentDB('api')
document_db_web3 = DocumentDB('web3')

# Default instance (web2)
document_db = document_db_web2

# Helper functions for each target
def get_db_web2(db_name):
    """Get a database from web2 DocumentDB"""
    return document_db_web2.get_database(db_name)

def get_db_api(db_name):
    """Get a database from api DocumentDB"""
    return document_db_api.get_database(db_name)

def get_db_web3(db_name):
    """Get a database from web3 DocumentDB"""
    return document_db_web3.get_database(db_name)

def get_db(db_name, target='web2'):
    """Get a database by name and target"""
    if target == 'web2':
        return document_db_web2.get_database(db_name)
    elif target == 'api':
        return document_db_api.get_database(db_name)
    elif target == 'web3':
        return document_db_web3.get_database(db_name)
    else:
        raise ValueError(f"Invalid target: {target}. Must be one of: web2, api, web3")

def list_all_databases(target='web2'):
    """List all databases for a specific target"""
    if target == 'web2':
        return document_db_web2.list_databases()
    elif target == 'api':
        return document_db_api.list_databases()
    elif target == 'web3':
        return document_db_web3.list_databases()
    else:
        raise ValueError(f"Invalid target: {target}. Must be one of: web2, api, web3")

def close_all_connections():
    """Close all database connections"""
    document_db_web2.close()
    document_db_api.close()
    document_db_web3.close()

# Test the connection when module is imported
if __name__ == "__main__":
    # Test the singleton pattern for different targets
    db1_web2 = DocumentDB('web2')
    print(db1_web2.get_client())
    # db2_web2 = DocumentDB('web2')
    # db1_api = DocumentDB('api')
    # db2_api = DocumentDB('api')

    # print("Creating indexes")
    # db1_web2.create_indexes(db1_web2.get_database('admin'))
    
    # print(f"Same web2 instance: {db1_web2 is db2_web2}")  # Should print True
    # print(f"Same api instance: {db1_api is db2_api}")  # Should print True
    # print(f"Different targets: {db1_web2 is db1_api}")  # Should print False
    
    # # Test connections for each target
    # targets = ['web2', 'api', 'web3']
    
    # for target in targets:
    #     try:
    #         print(f"\n{'='*50}")
    #         print(f"Testing {target} connection...")
    #         db = DocumentDB(target)
    #         db.connect()
    #         db.list_databases()
    #     except Exception as e:
    #         print(f"Failed to connect to {target}: {e}")
    #     finally:
    #         db.close()