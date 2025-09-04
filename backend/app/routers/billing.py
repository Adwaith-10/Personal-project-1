from fastapi import APIRouter, Depends, HTTPException, status
from app.models import User
from app.schemas import CheckoutSessionRequest, CheckoutSessionResponse
from app.deps import get_current_user
from app.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/create-checkout-session", response_model=CheckoutSessionResponse)
async def create_checkout_session(
    checkout_data: CheckoutSessionRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a Stripe checkout session for subscription"""
    if not settings.STRIPE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Billing is not enabled on this server"
        )
    
    try:
        import stripe
        
        # Set Stripe API key
        stripe.api_key = settings.STRIPE_SECRET_KEY
        
        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            customer_email=current_user.email,
            line_items=[
                {
                    "price": checkout_data.price_id,
                    "quantity": 1,
                }
            ],
            mode="subscription",
            success_url=checkout_data.success_url,
            cancel_url=checkout_data.cancel_url,
            metadata={
                "user_id": current_user.id,
                "user_email": current_user.email
            }
        )
        
        logger.info(f"✅ Checkout session created for user {current_user.email}")
        
        return CheckoutSessionResponse(
            session_id=checkout_session.id,
            checkout_url=checkout_session.url
        )
        
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Stripe package not installed"
        )
    except stripe.error.StripeError as e:
        logger.error(f"❌ Stripe error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Checkout session creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create checkout session"
        )


@router.get("/subscription-status")
async def get_subscription_status(
    current_user: User = Depends(get_current_user)
):
    """Get current user's subscription status"""
    if not settings.STRIPE_ENABLED:
        return {
            "billing_enabled": False,
            "subscription_status": "disabled"
        }
    
    try:
        import stripe
        
        # Set Stripe API key
        stripe.api_key = settings.STRIPE_SECRET_KEY
        
        # Get customer by email
        customers = stripe.Customer.list(email=current_user.email, limit=1)
        
        if not customers.data:
            return {
                "billing_enabled": True,
                "subscription_status": "no_subscription",
                "customer_id": None
            }
        
        customer = customers.data[0]
        
        # Get active subscriptions
        subscriptions = stripe.Subscription.list(
            customer=customer.id,
            status="active"
        )
        
        if subscriptions.data:
            subscription = subscriptions.data[0]
            return {
                "billing_enabled": True,
                "subscription_status": "active",
                "customer_id": customer.id,
                "subscription_id": subscription.id,
                "current_period_end": subscription.current_period_end,
                "plan": subscription.items.data[0].price.id if subscription.items.data else None
            }
        else:
            return {
                "billing_enabled": True,
                "subscription_status": "no_subscription",
                "customer_id": customer.id
            }
        
    except ImportError:
        return {
            "billing_enabled": False,
            "subscription_status": "stripe_not_installed"
        }
    except stripe.error.StripeError as e:
        logger.error(f"❌ Stripe error: {e}")
        return {
            "billing_enabled": True,
            "subscription_status": "error",
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"❌ Subscription status check failed: {e}")
        return {
            "billing_enabled": True,
            "subscription_status": "error",
            "error": "Unknown error"
        }


@router.post("/webhook")
async def stripe_webhook():
    """Handle Stripe webhooks for subscription events"""
    if not settings.STRIPE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Billing is not enabled on this server"
        )
    
    # This endpoint would handle webhook events from Stripe
    # Implementation depends on your specific needs
    # For now, return a placeholder response
    
    return {
        "message": "Webhook endpoint configured",
        "note": "Implement webhook handling based on your requirements"
    }


@router.get("/pricing")
async def get_pricing():
    """Get available pricing plans"""
    if not settings.STRIPE_ENABLED:
        return {
            "billing_enabled": False,
            "message": "Billing is not enabled on this server"
        }
    
    # Return pricing information
    # In production, you'd fetch this from Stripe or your database
    return {
        "billing_enabled": True,
        "plans": [
            {
                "id": "price_basic_monthly",
                "name": "Basic Plan",
                "price": 9.99,
                "currency": "usd",
                "interval": "month",
                "features": [
                    "Unlimited food analysis",
                    "Basic nutrition tracking",
                    "Mobile app access"
                ]
            },
            {
                "id": "price_premium_monthly",
                "name": "Premium Plan",
                "price": 19.99,
                "currency": "usd",
                "interval": "month",
                "features": [
                    "Everything in Basic",
                    "Advanced analytics",
                    "Personalized recommendations",
                    "Priority support"
                ]
            }
        ]
    }
