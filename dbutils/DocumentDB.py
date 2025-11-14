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
        'web2_documentdb': 'aiseer-documentdb-1.c3gsycya27j9.us-west-2.docdb.amazonaws.com',
        'api_documentdb': 'aiseer-documentdb-2.c3gsycya27j9.us-west-2.docdb.amazonaws.com',
        'web3_documentdb': 'aiseer-documentdb-3.c3gsycya27j9.us-west-2.docdb.amazonaws.com'
    }
    
    # Direct MongoDB connections (no SSH tunnel)
    WEB2_DIRECT_URI = "mongodb://prodadmin:u43ECwAf1j7RnqnP@ec2-54-201-211-20.us-west-2.compute.amazonaws.com:27017/?directConnection=true&retryWrites=false"
    WEB3_DIRECT_URI = "mongodb://prodadmin:u43ECwAf1j7RnqnP@ec2-54-213-32-246.us-west-2.compute.amazonaws.com:27017/?directConnection=true&retryWrites=false"
    API_DIRECT_URI = "mongodb://prodadmin:u43ECwAf1j7RnqnP@ec2-35-165-103-243.us-west-2.compute.amazonaws.com:27017/?directConnection=true&retryWrites=false"
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
            # For direct connections (web2, web3, api), DOCDB_HOST is not used, but we set a default
            # For SSH tunnel connections, use the appropriate endpoint
            self.DOCDB_HOST = self.DOCDB_ENDPOINTS.get(name, self.DOCDB_ENDPOINTS.get('web2_documentdb', None))
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
    
    def _log_pool_stats(self):
        """Log connection pool statistics for monitoring"""
        try:
            if self._client:
                # Get pool information from client
                nodes = list(self._client.nodes)
                if nodes:
                    print(f"🏊 Connection Pool Status:")
                    print(f"  • Active Nodes: {len(nodes)}")
                    print(f"  • Pool Configuration: Optimized for SSH tunnels")
                    print(f"  • Health Check: Connection validated ✅")
                else:
                    print("⚠️  No active nodes in connection pool")
        except Exception as e:
            print(f"⚠️  Could not retrieve pool stats: {e}")
    
    def check_connection_health(self):
        """Check if the database connection is healthy"""
        try:
            if not self._client:
                return False
            
            # Ping with timeout to check health
            result = self._client.admin.command('ping', maxTimeMS=1000)
            return result.get('ok') == 1
            
        except Exception as e:
            print(f"🏥 Connection health check failed: {e}")
            return False
    
    def get_pool_info(self):
        """Get detailed connection pool information"""
        try:
            if not self._client:
                return {"status": "no_client", "healthy": False}
            
            is_healthy = self.check_connection_health()
            nodes = list(self._client.nodes)
            
            return {
                "status": "connected" if is_healthy else "unhealthy",
                "healthy": is_healthy,
                "nodes_count": len(nodes),
                "tunnel_port": self._tunnel.local_bind_port if self._tunnel else None,
                "database_name": self.name
            }
            
        except Exception as e:
            return {"status": "error", "healthy": False, "error": str(e)}
    
    def get_available_port(self):
        """Get an available port dynamically to avoid conflicts"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
            print(f"🎯 Found available port: {port}")
            return port

    def connect(self):
        """Establish connection to DocumentDB (SSH tunnel for api, direct for web2/web3)"""
        if self._client is not None:
            print(f"🔄 Reusing existing connection to {self.name} DocumentDB")
            return self._client
        
        # Check if this is web2 or web3 with direct connection
        if self.name in ['web2', 'web3', 'api']:
            return self._connect_direct()
        else:
            return self._connect_ssh_tunnel()
    
    def _connect_direct(self):
        """Direct connection to MongoDB (for web3)"""
        try:
            print(f"🔌 Connecting directly to {self.name} MongoDB")
            print(f"📡 Direct connection using MongoDB URI")
            
            # Connection pool settings for direct connection
            pool_config = {
                # Connection Pool Optimization
                'maxPoolSize': 30,              # Higher for direct connection
                'minPoolSize': 5,               # Maintain minimum connections
                'maxIdleTimeMS': 120000,        # 2 minutes
                'maxConnecting': 10,            # More concurrent connections allowed
                
                # Timeout Optimization
                'serverSelectionTimeoutMS': 5000,  # 5s server selection
                'socketTimeoutMS': 20000,          # 20s socket timeout
                'connectTimeoutMS': 10000,         # 10s connection timeout
                'waitQueueTimeoutMS': 5000,        # 5s queue timeout
                
                # Health & Monitoring
                'heartbeatFrequencyMS': 30000,     # 30s heartbeat
                'localThresholdMS': 15,            # 15ms local threshold
                
                # Retry & Recovery
                'retryWrites': False,              # Disabled for MongoDB compatibility
                'retryReads': True,                # Enable read retries
            }
            
            print(f"📊 Connection Pool Settings (Direct):")
            print(f"  • Max Pool Size: {pool_config['maxPoolSize']}")
            print(f"  • Min Pool Size: {pool_config['minPoolSize']}")
            print(f"  • Max Idle Time: {pool_config['maxIdleTimeMS']/1000}s")
            print(f"  • Socket Timeout: {pool_config['socketTimeoutMS']/1000}s")
            
            # Select the appropriate URI based on database name
            if self.name == 'web2':
                connection_uri = self.WEB2_DIRECT_URI
            elif self.name == 'web3':
                connection_uri = self.WEB3_DIRECT_URI
            elif self.name == 'api':
                connection_uri = self.API_DIRECT_URI
            else:
                raise ValueError(f"Direct connection not configured for {self.name}")
            
            # Create MongoDB client with direct connection using URI
            self._client = MongoClient(
                connection_uri,
                **pool_config
            )
            
            # Test connection
            print(f"🧪 Testing connection to {self.name}...")
            server_info = self._client.server_info()
            print(f"✅ Connected successfully to {self.name}!")
            print(f"📊 Server version: {server_info.get('version', 'Unknown')}")
            
            # Display connection pool status
            self._log_pool_stats()
            
            # List all databases
            print(f"\n📂 Databases available on {self.name}:")
            db_names = self._client.list_database_names()
            for db_name in db_names:
                print(f"  - {db_name}")
            print(f"📊 Total databases: {len(db_names)}")
            
            return self._client
            
        except Exception as e:
            print(f"❌ Error connecting to {self.name}:", e)
            raise
    
    def _connect_ssh_tunnel(self):
        """Connection through SSH tunnel (for web2/api)"""
        try:
            print(f"🔌 Connecting to {self.name} DocumentDB: {self.DOCDB_HOST}")
            print(f"📡 Using SSH tunnel through {self.BASTION_HOST}")
            
            # ========== Dynamic Port Allocation ==========
            # Get an available port dynamically to avoid conflicts
            local_port = self.get_available_port()
            print(f"🔗 Using dynamic local port: {local_port}")
            
            # Retry SSH tunnel creation with different ports if needed
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        local_port = self.get_available_port()
                        print(f"🔄 Retry {attempt}: Using new port {local_port}")
                    
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
                    break  # Success, exit retry loop
                    
                except Exception as tunnel_error:
                    print(f"⚠️  SSH tunnel attempt {attempt + 1} failed: {tunnel_error}")
                    if self._tunnel:
                        try:
                            self._tunnel.stop()
                        except:
                            pass
                        self._tunnel = None
                    
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to establish SSH tunnel after {max_retries} attempts: {tunnel_error}")
                    
                    import time
                    time.sleep(1)  # Wait 1 second before retry
            
            print("🔐 Creating MongoDB client with optimized connection pool...")
            
            # Optimized connection pool settings for SSH tunnel + high concurrency
            pool_config = {
                # Connection Pool Optimization
                'maxPoolSize': 15,              # Reduced for SSH tunnel stability (was 30)
                'minPoolSize': 3,               # Maintain minimum connections (was 5)
                'maxIdleTimeMS': 60000,         # 1 minute - longer for better reuse (was 30s)
                'maxConnecting': 5,             # Limit concurrent connection attempts
                
                # Timeout Optimization
                'serverSelectionTimeoutMS': 3000,  # 3s - faster failure detection (was 5s)
                'socketTimeoutMS': 15000,           # 15s - longer for fact-checking operations (was 10s)
                'connectTimeoutMS': 8000,           # 8s connection timeout (was 10s)
                'waitQueueTimeoutMS': 3000,         # 3s queue timeout (was 5s)
                
                # Health & Monitoring
                'heartbeatFrequencyMS': 30000,      # 30s heartbeat for connection health
                'localThresholdMS': 15,             # 15ms local threshold for server selection
                
                # Retry & Recovery
                'retryWrites': False,               # Disable for DocumentDB compatibility
                'retryReads': True,                 # Enable read retries
                # Note: maxStalenessSeconds removed - conflicts with primary read preference in DocumentDB
                
                # TLS & Auth
                'tls': True,
                'tlsCAFile': self.CA_CERT_PATH,
                'authSource': 'admin',
                'tlsAllowInvalidHostnames': True,
                'directConnection': True,
            }
            
            print(f"📊 Connection Pool Settings:")
            print(f"  • Max Pool Size: {pool_config['maxPoolSize']}")
            print(f"  • Min Pool Size: {pool_config['minPoolSize']}")
            print(f"  • Max Idle Time: {pool_config['maxIdleTimeMS']/1000}s")
            print(f"  • Socket Timeout: {pool_config['socketTimeoutMS']/1000}s")
            print(f"  • Connection Timeout: {pool_config['connectTimeoutMS']/1000}s")
            
            self._client = MongoClient(
                host='localhost',
                port=self._tunnel.local_bind_port,
                username=self.DB_USERNAME,
                password=self.DB_PASSWORD,
                **pool_config
            )

            # Test connection and validate pool
            print(f"🧪 Testing connection to {self.name}...")
            server_info = self._client.server_info()
            print(f"✅ Connected successfully to {self.name}!")
            print(f"📊 Server version: {server_info.get('version', 'Unknown')}")
            print(f"🔧 Server features: {server_info.get('features', {})}")
            
            # Display connection pool status
            self._log_pool_stats()

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
        
        # 8. Credit-related collections (Performance optimization)
        print("  📊 Creating indexes for credit collections...")
        
        # User daily credits - most frequently queried
        try:
            db.user_daily_credits.create_index([("userEmail", ASCENDING)], background=True)
            db.user_daily_credits.create_index([("userEmail", ASCENDING), ("lastUpdated", ASCENDING)], background=True)
            print("    ✅ user_daily_credits indexes created")
        except Exception as e:
            print(f"    ⚠️  user_daily_credits indexes: {e}")
        
        # User bonus credits
        try:
            db.user_bonus_credits.create_index([("user_email", ASCENDING)], background=True)
            print("    ✅ user_bonus_credits indexes created")
        except Exception as e:
            print(f"    ⚠️  user_bonus_credits indexes: {e}")
        
        # User additional credits
        try:
            db.user_additional_credits.create_index([("userEmail", ASCENDING)], background=True)
            print("    ✅ user_additional_credits indexes created")
        except Exception as e:
            print(f"    ⚠️  user_additional_credits indexes: {e}")
        
        # Web3 users mapping
        try:
            db.web3_users.create_index([("user_id", ASCENDING)], background=True)
            db.web3_users.create_index([("wallet_id", ASCENDING)], background=True)
            print("    ✅ web3_users indexes created")
        except Exception as e:
            print(f"    ⚠️  web3_users indexes: {e}")
        
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
document_db_web2_ssh = DocumentDB('web2_documentdb')

# Default instance (web2)``
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