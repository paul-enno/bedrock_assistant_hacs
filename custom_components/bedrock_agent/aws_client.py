"""AWS client factory for Bedrock integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


class AWSClientFactory:
    """Factory for creating AWS clients with proper async handling."""

    def __init__(
        self,
        hass: HomeAssistant,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
    ) -> None:
        """Initialize the AWS client factory."""
        self.hass = hass
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name

    def create_boto3_session(self) -> boto3.Session:
        """Create a boto3 session (synchronous, for use in executor)."""
        return boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )
