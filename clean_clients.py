import etcd3

def clean():
    try:
        etcd = etcd3.client(host='127.0.0.1', port=2379)
        # Delete all keys starting with /clients/
        etcd.delete_prefix('/clients/')
        print("✅ Successfully cleared stale clients from etcd.")
    except Exception as e:
        print(f"❌ Error clearing etcd: {e}")

if __name__ == "__main__":
    clean()