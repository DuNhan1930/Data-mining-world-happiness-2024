package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.preprocessing;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AddColumn {

    private static final Map<String, String> COUNTRY_TO_REGION = new HashMap<>();

    static {
        // --- Africa ---
        String africa = "Africa";
        for (String c : new String[]{
                "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
                "Central African Republic", "Chad", "Comoros", "Congo (Brazzaville)", "Congo (Kinshasa)",
                "Djibouti", "Egypt", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
                "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali",
                "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
                "Rwanda", "Senegal", "Sierra Leone", "Somalia", "Somaliland region", "South Africa",
                "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
        }) { COUNTRY_TO_REGION.put(c, africa); }

        // --- North America ---
        String na = "North America";
        for (String c : new String[]{
                "Belize", "Canada", "Costa Rica", "Dominican Republic", "El Salvador", "Guatemala",
                "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Trinidad and Tobago",
                "United States"
        }) { COUNTRY_TO_REGION.put(c, na); }

        // --- South America ---
        String sa = "South America";
        for (String c : new String[]{
                "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana",
                "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"
        }) { COUNTRY_TO_REGION.put(c, sa); }

        // --- Central Asia ---
        String ca = "Central Asia";
        for (String c : new String[]{
                "Kazakhstan", "Kyrgyzstan", "Tajikistan", "Turkmenistan", "Uzbekistan"
        }) { COUNTRY_TO_REGION.put(c, ca); }

        // --- East Asia ---
        String ea = "East Asia";
        for (String c : new String[]{
                "China", "Hong Kong S.A.R. of China", "Japan", "Mongolia", "South Korea",
                "Taiwan Province of China"
        }) { COUNTRY_TO_REGION.put(c, ea); }

        // --- South Asia ---
        String sas = "South Asia";
        for (String c : new String[]{
                "Afghanistan", "Bangladesh", "Bhutan", "India", "Maldives", "Nepal", "Pakistan", "Sri Lanka"
        }) { COUNTRY_TO_REGION.put(c, sas); }

        // --- Southeast Asia ---
        String sea = "Southeast Asia";
        for (String c : new String[]{
                "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore",
                "Thailand", "Vietnam"
        }) { COUNTRY_TO_REGION.put(c, sea); }

        // --- Western Asia ---
        String wa = "Western Asia";
        for (String c : new String[]{
                "Armenia", "Azerbaijan", "Bahrain", "Cyprus", "Georgia", "Iran", "Iraq", "Israel",
                "Jordan", "Kuwait", "Lebanon", "Oman", "Qatar", "Saudi Arabia", "State of Palestine",
                "Syria", "Türkiye", "United Arab Emirates", "Yemen"
        }) { COUNTRY_TO_REGION.put(c, wa); }

        // --- Europe ---
        String eu = "Europe";
        for (String c : new String[]{
                "Albania", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
                "Croatia", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
                "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", "Lithuania", "Luxembourg",
                "Malta", "Moldova", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland",
                "Portugal", "Romania", "Russia", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden",
                "Switzerland", "Ukraine", "United Kingdom"
        }) { COUNTRY_TO_REGION.put(c, eu); }

        // --- Oceania ---
        String oc = "Oceania";
        for (String c : new String[]{
                "Australia", "New Zealand"
        }) { COUNTRY_TO_REGION.put(c, oc); }
    }

    /**
     * Adds a "Region" column based on the "Country name" column.
     * @param data The Weka Instances object.
     * @param countryColName The name of the country attribute (e.g., "Country name").
     * @param newRegionColName The name of the new attribute (e.g., "Region").
     */
    public static void addRegionBasedOnCountry(Instances data, String countryColName, String newRegionColName) {
        Attribute countryAttr = data.attribute(countryColName);
        if (countryAttr == null) {
            System.out.println("Warning: Country column '" + countryColName + "' not found. Cannot add Region.");
            return;
        }

        // 1. Define the nominal values for the new Region attribute
        ArrayList<String> regionValues = new ArrayList<>();
        regionValues.add("Africa");
        regionValues.add("North America");
        regionValues.add("South America");
        regionValues.add("Central Asia");
        regionValues.add("East Asia");
        regionValues.add("South Asia");
        regionValues.add("Southeast Asia");
        regionValues.add("Western Asia");
        regionValues.add("Europe");
        regionValues.add("Oceania");
        regionValues.add("Other"); // Fallback

        // 2. Create and insert the new Attribute
        Attribute regionAttr = new Attribute(newRegionColName, regionValues);
        data.insertAttributeAt(regionAttr, data.numAttributes()); // Add at end
        int regionIdx = data.numAttributes() - 1;
        int countryIdx = countryAttr.index();

        // 3. Iterate and set values
        int mappedCount = 0;
        int missingCount = 0;

        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            if (inst.isMissing(countryIdx)) {
                inst.setMissing(regionIdx);
                continue;
            }

            String country = inst.stringValue(countryIdx);
            String region = COUNTRY_TO_REGION.get(country);

            if (region != null) {
                inst.setValue(regionIdx, region);
                mappedCount++;
            } else {
                // If country not found in our map
                inst.setValue(regionIdx, "Other");
                System.out.println("Warning: Region not found for country: " + country);
                missingCount++;
            }
        }

        System.out.println("Added '" + newRegionColName + "' column. Mapped " + mappedCount + " instances. Unmapped/Other: " + missingCount);
    }
}
