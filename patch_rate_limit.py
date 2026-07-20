import re

with open("worker/utils/rate_limit.py", "r") as f:
    content = f.read()

# Replace the fallback that reads from COSTS_FILE
pattern1 = re.compile(
    r'        else:\n'
    r'            # Fallback to checking costs\.json\n'
    r'            if os\.path\.exists\(COSTS_FILE\):\n'
    r'                with open\(COSTS_FILE\) as f:\n'
    r'                    persisted = json\.load\(f\)\n'
    r'                    if model_lower in persisted:\n'
    r'                        in_rate = float\(persisted\[model_lower\]\.get\("prompt", 0\)\)\n'
    r'                        out_rate = float\(persisted\[model_lower\]\.get\("completion", 0\)\)',
    re.DOTALL
)

new1 = """        else:
            # Removed costs.json fallback as part of Phase E.3
            pass"""

content, count1 = pattern1.subn(new1, content)
print(f"Replaced {count1} fallbacks.")

# Remove COSTS_FILE read/write in update_model_costs
pattern2 = re.compile(
    r'    # Load existing costs\n'
    r'    persisted_costs = \{\}\n'
    r'    if os\.path\.exists\(COSTS_FILE\):\n'
    r'        try:\n'
    r'            with open\(COSTS_FILE\) as f:\n'
    r'                persisted_costs = json\.load\(f\)\n'
    r'        except Exception as e:\n'
    r'            logger\.warning\(f"Failed to read \{COSTS_FILE\}: \{e\}"\)',
    re.DOTALL
)

new2 = """    # Load existing costs (now only from Redis, local file deprecated Phase E.3)
    persisted_costs = {}
    try:
        keys = redis_client.keys("model_cost:*")
        for key in keys:
            model = key.decode("utf-8").split(":", 1)[1]
            data = redis_client.get(key)
            if data:
                parsed = json.loads(data)
                # Keep timestamp hack by using current time if it's cached in redis
                parsed["timestamp"] = time.time()
                persisted_costs[model] = parsed
    except Exception as e:
        logger.warning(f"Failed to read from Redis: {e}")"""

content, count2 = pattern2.subn(new2, content)
print(f"Replaced {count2} read logic.")

pattern3 = re.compile(
    r'    finally:\n'
    r'        # Save persisted costs\n'
    r'        try:\n'
    r'            with open\(COSTS_FILE, "w"\) as f:\n'
    r'                json\.dump\(persisted_costs, f, indent=2\)\n'
    r'        except Exception as e:\n'
    r'            logger\.warning\(f"Failed to write \{COSTS_FILE\}: \{e\}"\)',
    re.DOTALL
)

new3 = """    finally:
        # Saving to costs.json removed as part of Phase E.3
        pass"""

content, count3 = pattern3.subn(new3, content)
print(f"Replaced {count3} write logic.")

with open("worker/utils/rate_limit.py", "w") as f:
    f.write(content)
