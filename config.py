#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# ss under the MIT License.

import os


class DefaultConfig:
    """Bot Configuration"""

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "fb527c81-09ec-4b44-a474-ba3e77709bdd")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "-dm8Q~aQZvZZvLd8WhjETdTgXxMLBWKKGrNfDaJP")
