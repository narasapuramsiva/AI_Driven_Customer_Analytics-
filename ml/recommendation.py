def generate_recommendations(df):
    """
    Generate business recommendations
    based on cluster behavior
    """

    recommendations = {}

    for cluster in df["Cluster"].unique():

        cluster_data = df[df["Cluster"] == cluster]

        avg_income = cluster_data["Income"].mean()
        avg_spending = cluster_data["SpendingScore"].mean()

        if avg_income > 70000 and avg_spending > 70:

            recommendations[cluster] = {
                "Customer Type": "Premium Customers",
                "Strategy": "Offer VIP Membership, Premium Products and Loyalty Rewards"
            }

        elif avg_income > 60000 and avg_spending < 50:

            recommendations[cluster] = {
                "Customer Type": "Potential Customers",
                "Strategy": "Provide Personalized Discounts and Cross-Selling Offers"
            }

        elif avg_income < 50000 and avg_spending > 70:

            recommendations[cluster] = {
                "Customer Type": "Frequent Buyers",
                "Strategy": "Offer Bundle Deals and Cashback Programs"
            }

        else:

            recommendations[cluster] = {
                "Customer Type": "Regular Customers",
                "Strategy": "Maintain Engagement through Email Campaigns and Seasonal Offers"
            }

    return recommendations